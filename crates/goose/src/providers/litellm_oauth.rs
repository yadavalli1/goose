use crate::config::paths::Paths;
use crate::config::Config;
use crate::providers::api_client::{ApiClient, AuthProvider, TlsConfig};
use crate::providers::oauth_device_flow::{
    poll_for_tokens, request_device_code, run_device_flow, DeviceCodeResponse, DeviceFlowConfig,
    RequestEncoding,
};
use crate::providers::private_file::write_private_file;
use crate::utils::bytes_to_hex;
use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use axum::{extract::Query, response::Html, routing::get, Router};
use base64::Engine;
use chrono::{DateTime, Duration, Utc};
use once_cell::sync::Lazy;
use reqwest::{header::HeaderMap, redirect::Policy, Client};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;
use std::{collections::HashMap, fs, net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::sync::{oneshot, Mutex as TokioMutex};
use url::{Host, Url};

static LITELLM_OAUTH_MUTEX: Lazy<TokioMutex<()>> = Lazy::new(|| TokioMutex::new(()));

const DEFAULT_OIDC_SCOPES: &[&str] = &["openid", "profile", "email", "offline_access"];
const DEFAULT_REDIRECT_URL: &str = "http://localhost";
const OAUTH_BROWSER_TIMEOUT_SECS: u64 = 120;
const TOKEN_REFRESH_SKEW_SECS: i64 = 60;
const DEFAULT_DEVICE_POLL_INTERVAL_SECS: u64 = 5;
const DEFAULT_DEVICE_EXPIRES_IN_SECS: u64 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OidcFlowMode {
    #[default]
    Auto,
    Browser,
    Device,
}

impl OidcFlowMode {
    pub fn from_str_opt(value: Option<&str>) -> Result<Self> {
        match value.map(str::trim).filter(|s| !s.is_empty()) {
            None => Ok(Self::Auto),
            Some(v) if v.eq_ignore_ascii_case("auto") => Ok(Self::Auto),
            Some(v)
                if v.eq_ignore_ascii_case("browser")
                    || v.eq_ignore_ascii_case("pkce")
                    || v.eq_ignore_ascii_case("authorization_code") =>
            {
                Ok(Self::Browser)
            }
            Some(v)
                if v.eq_ignore_ascii_case("device") || v.eq_ignore_ascii_case("device_code") =>
            {
                Ok(Self::Device)
            }
            Some(other) => bail!(
                "Invalid LITELLM_OIDC_FLOW '{}': expected 'auto', 'browser', or 'device'",
                other
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OidcTokenType {
    #[default]
    AccessToken,
    IdToken,
}

impl OidcTokenType {
    pub fn from_str_opt(value: Option<&str>) -> Result<Self> {
        match value.map(str::trim).filter(|s| !s.is_empty()) {
            None => Ok(Self::AccessToken),
            Some(v)
                if v.eq_ignore_ascii_case("access_token") || v.eq_ignore_ascii_case("access") =>
            {
                Ok(Self::AccessToken)
            }
            Some(v) if v.eq_ignore_ascii_case("id_token") || v.eq_ignore_ascii_case("id") => {
                Ok(Self::IdToken)
            }
            Some(other) => bail!(
                "Invalid LITELLM_OIDC_TOKEN_TYPE '{}': expected 'access_token' or 'id_token'",
                other
            ),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AccessToken => "access_token",
            Self::IdToken => "id_token",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LiteLLMOidcConfig {
    pub issuer_url: String,
    pub client_id: String,
    pub client_secret: Option<String>,
    pub scopes: Vec<String>,
    pub audience: Option<String>,
    pub flow: OidcFlowMode,
    pub redirect_url: String,
    pub token_type: OidcTokenType,
}

impl LiteLLMOidcConfig {
    pub fn from_config(config: &Config) -> Result<Option<Self>> {
        let issuer_url = config
            .get_param::<String>("LITELLM_OIDC_ISSUER_URL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let client_id = config
            .get_param::<String>("LITELLM_OIDC_CLIENT_ID")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let (issuer_url, client_id) = match (issuer_url, client_id) {
            (Some(issuer), Some(cid)) => (issuer, cid),
            (None, None) => return Ok(None),
            (Some(_), None) => {
                bail!("LITELLM_OIDC_CLIENT_ID is required when LITELLM_OIDC_ISSUER_URL is set")
            }
            (None, Some(_)) => {
                bail!("LITELLM_OIDC_ISSUER_URL is required when LITELLM_OIDC_CLIENT_ID is set")
            }
        };

        let client_secret = config
            .get_secret::<String>("LITELLM_OIDC_CLIENT_SECRET")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let scopes = config
            .get_param::<String>("LITELLM_OIDC_SCOPES")
            .ok()
            .map(|raw| parse_scopes(&raw))
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| {
                DEFAULT_OIDC_SCOPES
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect()
            });

        let audience = config
            .get_param::<String>("LITELLM_OIDC_AUDIENCE")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let flow_raw = config.get_param::<String>("LITELLM_OIDC_FLOW").ok();
        let flow = OidcFlowMode::from_str_opt(flow_raw.as_deref())?;

        let redirect_url = config
            .get_param::<String>("LITELLM_OIDC_REDIRECT_URL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_REDIRECT_URL.to_string());

        let token_type_raw = config.get_param::<String>("LITELLM_OIDC_TOKEN_TYPE").ok();
        let token_type = OidcTokenType::from_str_opt(token_type_raw.as_deref())?;

        Ok(Some(Self {
            issuer_url,
            client_id,
            client_secret,
            scopes,
            audience,
            flow,
            redirect_url,
            token_type,
        }))
    }
}

fn parse_scopes(raw: &str) -> Vec<String> {
    raw.split(|c: char| c == ',' || c.is_whitespace())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

#[derive(Debug, Clone)]
pub struct OidcEndpoints {
    pub authorization_endpoint: Option<String>,
    pub token_endpoint: String,
    pub device_authorization_endpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenData {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
}

impl TokenData {
    pub fn select_jwt(&self, token_type: OidcTokenType) -> Result<String> {
        match token_type {
            OidcTokenType::AccessToken => Ok(self.access_token.clone()),
            OidcTokenType::IdToken => self.id_token.clone().ok_or_else(|| {
                anyhow!(
                    "LITELLM_OIDC_TOKEN_TYPE is set to 'id_token', but the OIDC provider did not return an id_token (ensure 'openid' scope is included)"
                )
            }),
        }
    }

    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(expires_at) => {
                expires_at <= Utc::now() + Duration::seconds(TOKEN_REFRESH_SKEW_SECS)
            }
            None => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenCache {
    cache_path: PathBuf,
}

fn get_base_path() -> PathBuf {
    Paths::in_config_dir("litellm/oauth")
}

pub fn cleanup_litellm_oauth_cache() -> Result<()> {
    let base_path = get_base_path();
    match fs::remove_dir_all(&base_path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

impl TokenCache {
    pub fn new(config: &LiteLLMOidcConfig) -> Self {
        Self::with_base_dir(get_base_path(), config)
    }

    pub fn with_base_dir(base_dir: PathBuf, config: &LiteLLMOidcConfig) -> Self {
        let mut hasher = sha2::Sha256::new();
        hasher.update(config.issuer_url.as_bytes());
        hasher.update(b"|");
        hasher.update(config.client_id.as_bytes());
        hasher.update(b"|");
        hasher.update(config.scopes.join(" ").as_bytes());
        hasher.update(b"|");
        if let Some(audience) = &config.audience {
            hasher.update(audience.as_bytes());
        }
        hasher.update(b"|");
        hasher.update(config.token_type.as_str().as_bytes());
        let hash = bytes_to_hex(hasher.finalize());

        let _ = fs::create_dir_all(&base_dir);
        let cache_path = base_dir.join(format!("{}.json", hash));
        Self { cache_path }
    }

    pub fn load(&self) -> Option<TokenData> {
        let contents = fs::read_to_string(&self.cache_path).ok()?;
        serde_json::from_str(&contents).ok()
    }

    pub fn save(&self, token_data: &TokenData) -> Result<()> {
        let contents = serde_json::to_string(token_data)?;
        write_private_file(&self.cache_path, &contents)?;
        Ok(())
    }

    pub fn clear(&self) {
        let _ = fs::remove_file(&self.cache_path);
    }
}

#[derive(Clone, Copy)]
enum EndpointTransport {
    Https,
    LoopbackHttp,
}

fn endpoint_transport(url: &Url) -> Result<EndpointTransport> {
    if url.scheme() == "https" {
        return Ok(EndpointTransport::Https);
    }

    let is_loopback = match url.host() {
        Some(Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(Host::Ipv4(address)) => address.is_loopback(),
        Some(Host::Ipv6(address)) => address.is_loopback(),
        None => false,
    };

    if url.scheme() == "http" && is_loopback {
        Ok(EndpointTransport::LoopbackHttp)
    } else {
        bail!(
            "OIDC endpoint '{}' must use HTTPS unless it targets loopback",
            url
        )
    }
}

fn build_http_client(endpoint: &Url, tls_config: Option<&TlsConfig>) -> Result<Client> {
    let transport = endpoint_transport(endpoint)?;
    if matches!(transport, EndpointTransport::Https) {
        if let Some(tls) = tls_config {
            if tls.is_configured() {
                return ApiClient::http_client(Some(tls));
            }
        }
        return Ok(Client::builder().https_only(true).build()?);
    }

    Ok(Client::builder()
        .no_proxy()
        .redirect(Policy::custom(|attempt| {
            if attempt.previous().len() >= 10 {
                return attempt.error("too many redirects");
            }
            if endpoint_transport(attempt.url()).is_ok() {
                attempt.follow()
            } else {
                attempt.error("redirect violates the OIDC endpoint transport policy")
            }
        }))
        .build()?)
}

fn discovery_urls(issuer: &str) -> Result<Vec<Url>> {
    let trimmed = issuer.trim();
    if trimmed.ends_with("/.well-known/openid-configuration")
        || trimmed.ends_with("/.well-known/oauth-authorization-server")
    {
        return Ok(vec![Url::parse(trimmed)?]);
    }

    let mut base = Url::parse(trimmed)?;
    let path = base.path().trim_end_matches('/').to_string();
    base.set_path(&format!("{}/", path));

    let openid = base.join(".well-known/openid-configuration")?;
    let oauth = base.join(".well-known/oauth-authorization-server")?;
    Ok(vec![openid, oauth])
}

pub async fn discover_oidc_endpoints(
    issuer: &str,
    tls_config: Option<&TlsConfig>,
) -> Result<OidcEndpoints> {
    let candidate_urls = discovery_urls(issuer)?;
    let mut last_error = None;

    for url in candidate_urls {
        let client = build_http_client(&url, tls_config)?;
        match client.get(url.clone()).send().await {
            Ok(resp) if resp.status().is_success() => {
                let doc: Value = resp
                    .json()
                    .await
                    .context("failed to parse OIDC discovery document")?;
                let token_endpoint = doc
                    .get("token_endpoint")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("token_endpoint not found in OIDC configuration"))?
                    .to_string();
                let token_url = Url::parse(&token_endpoint)?;
                endpoint_transport(&token_url)?;

                let authorization_endpoint = doc
                    .get("authorization_endpoint")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                if let Some(auth_ep) = &authorization_endpoint {
                    let auth_url = Url::parse(auth_ep)?;
                    endpoint_transport(&auth_url)?;
                }

                let device_authorization_endpoint = doc
                    .get("device_authorization_endpoint")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                if let Some(dev_ep) = &device_authorization_endpoint {
                    let dev_url = Url::parse(dev_ep)?;
                    endpoint_transport(&dev_url)?;
                }

                return Ok(OidcEndpoints {
                    authorization_endpoint,
                    token_endpoint,
                    device_authorization_endpoint,
                });
            }
            Ok(resp) => {
                last_error = Some(anyhow!(
                    "OIDC discovery at {} returned HTTP {}",
                    url,
                    resp.status()
                ));
            }
            Err(e) => {
                last_error = Some(anyhow!("OIDC discovery request to {} failed: {}", url, e));
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("Failed to discover OIDC endpoints for {}", issuer)))
}

fn extract_token_data(
    token_response: &Value,
    previous_refresh_token: Option<&str>,
    previous_id_token: Option<&str>,
) -> Result<TokenData> {
    let access_token = token_response
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("access_token not found in OIDC token response"))?
        .to_string();

    let id_token = token_response
        .get("id_token")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| previous_id_token.map(ToString::to_string));

    let refresh_token = token_response
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| previous_refresh_token.map(ToString::to_string));

    let expires_at = token_response
        .get("expires_in")
        .and_then(Value::as_i64)
        .map(|secs| Utc::now() + Duration::seconds(secs));

    Ok(TokenData {
        access_token,
        id_token,
        refresh_token,
        expires_at,
    })
}

async fn exchange_code_for_token(
    endpoints: &OidcEndpoints,
    config: &LiteLLMOidcConfig,
    code: &str,
    redirect_url: &str,
    verifier: &str,
    tls_config: Option<&TlsConfig>,
) -> Result<TokenData> {
    let token_url = Url::parse(&endpoints.token_endpoint)?;
    let client = build_http_client(&token_url, tls_config)?;

    let mut params = vec![
        ("grant_type", "authorization_code".to_string()),
        ("code", code.to_string()),
        ("redirect_uri", redirect_url.to_string()),
        ("code_verifier", verifier.to_string()),
        ("client_id", config.client_id.clone()),
    ];
    if let Some(secret) = &config.client_secret {
        params.push(("client_secret", secret.clone()));
    }

    let resp = client
        .post(token_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&params)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let err_text = resp.text().await.unwrap_or_default();
        bail!(
            "Failed to exchange authorization code for token ({}): {}",
            status,
            err_text
        );
    }

    let token_response: Value = resp.json().await?;
    extract_token_data(&token_response, None, None)
}

pub async fn refresh_oidc_token(
    endpoints: &OidcEndpoints,
    config: &LiteLLMOidcConfig,
    refresh_token: &str,
    previous_id_token: Option<&str>,
    tls_config: Option<&TlsConfig>,
) -> Result<TokenData> {
    let token_url = Url::parse(&endpoints.token_endpoint)?;
    let client = build_http_client(&token_url, tls_config)?;

    let mut params = vec![
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", refresh_token.to_string()),
        ("client_id", config.client_id.clone()),
    ];
    if let Some(secret) = &config.client_secret {
        params.push(("client_secret", secret.clone()));
    }

    let resp = client
        .post(token_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&params)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let err_text = resp.text().await.unwrap_or_default();
        bail!("Failed to refresh OIDC token ({}): {}", status, err_text);
    }

    let token_response: Value = resp.json().await?;
    extract_token_data(&token_response, Some(refresh_token), previous_id_token)
}

pub fn build_authorization_url(
    authorization_endpoint: &str,
    config: &LiteLLMOidcConfig,
    redirect_url: &str,
    state: &str,
    verifier: &str,
) -> Result<String> {
    let digest = sha2::Sha256::digest(verifier.as_bytes());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
    let scope_str = config.scopes.join(" ");

    let mut url = Url::parse(authorization_endpoint)?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("response_type", "code");
        query.append_pair("client_id", &config.client_id);
        query.append_pair("redirect_uri", redirect_url);
        query.append_pair("scope", &scope_str);
        query.append_pair("state", state);
        query.append_pair("code_challenge", &challenge);
        query.append_pair("code_challenge_method", "S256");
        if let Some(audience) = &config.audience {
            query.append_pair("audience", audience);
        }
    }
    Ok(url.to_string())
}

async fn run_browser_pkce_flow(
    endpoints: &OidcEndpoints,
    config: &LiteLLMOidcConfig,
    tls_config: Option<&TlsConfig>,
) -> Result<TokenData> {
    let authorization_endpoint = endpoints.authorization_endpoint.as_deref().ok_or_else(|| {
        anyhow!("authorization_endpoint not found in OIDC configuration for browser flow")
    })?;

    let state = nanoid::nanoid!(16);
    let verifier = nanoid::nanoid!(64);

    let parsed_redirect =
        Url::parse(&config.redirect_url).context("Invalid LITELLM_OIDC_REDIRECT_URL")?;
    let callback_path = if parsed_redirect.path().is_empty() {
        "/".to_string()
    } else {
        parsed_redirect.path().to_string()
    };

    let (tx, rx) = oneshot::channel::<Result<String>>();
    let tx = Arc::new(TokioMutex::new(Some(tx)));
    let expected_state = state.clone();

    let app = Router::new().route(
        &callback_path,
        get(move |Query(params): Query<HashMap<String, String>>| {
            let tx = Arc::clone(&tx);
            let expected_state = expected_state.clone();
            async move {
                if let Some(err) = params.get("error") {
                    let desc = params
                        .get("error_description")
                        .cloned()
                        .unwrap_or_else(|| err.clone());
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err(anyhow!("OIDC error: {}", desc)));
                    }
                    return Html("<h2>Authentication Error</h2><p>You can close this window.</p>");
                }

                let code = params.get("code").cloned();
                let received_state = params.get("state").cloned();

                if let (Some(code), Some(received_state)) = (code, received_state) {
                    if received_state == expected_state {
                        if let Some(sender) = tx.lock().await.take() {
                            if sender.send(Ok(code)).is_ok() {
                                return Html(
                                    "<h2>Login Success</h2><p>You can close this window and return to goose.</p>",
                                );
                            }
                        }
                        Html("<h2>Error</h2><p>Authentication already completed.</p>")
                    } else {
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(Err(anyhow!("OIDC state mismatch")));
                        }
                        Html("<h2>Error</h2><p>State mismatch.</p>")
                    }
                } else {
                    Html("<h2>Error</h2><p>Authentication failed.</p>")
                }
            }
        }),
    );

    let env_port: Option<u16> = std::env::var("GOOSE_OAUTH_CALLBACK_PORT")
        .ok()
        .and_then(|p| p.parse().ok());
    let bind_port = parsed_redirect.port().or(env_port).unwrap_or(0);
    let addr = SocketAddr::from(([127, 0, 0, 1], bind_port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let actual_port = listener.local_addr()?.port();

    let server_handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let host_str = parsed_redirect.host_str().unwrap_or("localhost");
    let path_suffix = if callback_path == "/" {
        ""
    } else {
        callback_path.as_str()
    };
    let actual_redirect_url = format!(
        "{}://{}:{}{}",
        parsed_redirect.scheme(),
        host_str,
        actual_port,
        path_suffix
    );

    let authorization_url = build_authorization_url(
        authorization_endpoint,
        config,
        &actual_redirect_url,
        &state,
        &verifier,
    )?;

    if webbrowser::open(&authorization_url).is_err() {
        eprintln!(
            "Please open this URL in your browser to authenticate with LiteLLM:\n{}",
            authorization_url
        );
    }

    let code_result = tokio::time::timeout(
        std::time::Duration::from_secs(OAUTH_BROWSER_TIMEOUT_SECS),
        rx,
    )
    .await;

    server_handle.abort();

    let code = code_result
        .map_err(|_| anyhow!("LiteLLM OIDC browser authentication timed out"))??
        .map_err(|e| anyhow!("LiteLLM OIDC callback failed: {}", e))?;

    exchange_code_for_token(
        endpoints,
        config,
        &code,
        &actual_redirect_url,
        &verifier,
        tls_config,
    )
    .await
}

pub async fn run_device_code_oidc_flow(
    endpoints: &OidcEndpoints,
    config: &LiteLLMOidcConfig,
    tls_config: Option<&TlsConfig>,
) -> Result<TokenData> {
    let device_auth_endpoint = endpoints
        .device_authorization_endpoint
        .as_deref()
        .ok_or_else(|| anyhow!("device_authorization_endpoint not found in OIDC configuration"))?;

    let device_url = Url::parse(device_auth_endpoint)?;
    let client = build_http_client(&device_url, tls_config)?;
    let scope_str = config.scopes.join(" ");

    if config.audience.is_none()
        && config.client_secret.is_none()
        && config.token_type == OidcTokenType::AccessToken
    {
        let df_cfg = DeviceFlowConfig {
            device_auth_url: Some(device_auth_endpoint),
            token_url: &endpoints.token_endpoint,
            client_id: &config.client_id,
            scopes: Some(&scope_str),
            extra_headers: HeaderMap::new(),
            encoding: RequestEncoding::Form,
        };
        let tokens = run_device_flow(&client, &df_cfg).await?;
        return Ok(TokenData {
            access_token: tokens.access_token,
            id_token: None,
            refresh_token: tokens.refresh_token,
            expires_at: tokens.expires_at,
        });
    }

    let mut form_params = vec![
        ("client_id", config.client_id.clone()),
        ("scope", scope_str.clone()),
    ];
    if let Some(audience) = &config.audience {
        form_params.push(("audience", audience.clone()));
    }
    if let Some(secret) = &config.client_secret {
        form_params.push(("client_secret", secret.clone()));
    }

    let device: DeviceCodeResponse = if config.audience.is_none() && config.client_secret.is_none()
    {
        let df_cfg = DeviceFlowConfig {
            device_auth_url: Some(device_auth_endpoint),
            token_url: &endpoints.token_endpoint,
            client_id: &config.client_id,
            scopes: Some(&scope_str),
            extra_headers: HeaderMap::new(),
            encoding: RequestEncoding::Form,
        };
        request_device_code(&client, &df_cfg).await?
    } else {
        client
            .post(device_url)
            .form(&form_params)
            .send()
            .await
            .context("failed to request OIDC device authorization")?
            .error_for_status()
            .context("OIDC device authorization request failed")?
            .json::<DeviceCodeResponse>()
            .await
            .context("failed to parse OIDC device authorization response")?
    };

    let verify_url = device.verification_url().to_string();
    let expires_in = device.expires_in.unwrap_or(DEFAULT_DEVICE_EXPIRES_IN_SECS);
    let interval = device.interval.unwrap_or(DEFAULT_DEVICE_POLL_INTERVAL_SECS);

    let copied = arboard::Clipboard::new()
        .ok()
        .and_then(|mut cb| cb.set_text(&device.user_code).ok())
        .is_some();
    if let Err(e) = webbrowser::open(&verify_url) {
        tracing::warn!("Failed to open browser: {}", e);
    }
    let clipboard_hint = if copied { " (copied to clipboard)" } else { "" };
    eprintln!(
        "Please visit {} and enter code {}{}",
        verify_url, device.user_code, clipboard_hint
    );

    poll_oidc_device_token(
        &client,
        &endpoints.token_endpoint,
        config,
        &device.device_code,
        interval,
        expires_in,
    )
    .await
}

async fn poll_oidc_device_token(
    client: &Client,
    token_endpoint: &str,
    config: &LiteLLMOidcConfig,
    device_code: &str,
    interval_secs: u64,
    expires_in_secs: u64,
) -> Result<TokenData> {
    if config.client_secret.is_none() && config.token_type == OidcTokenType::AccessToken {
        let scope_str = config.scopes.join(" ");
        let df_cfg = DeviceFlowConfig {
            device_auth_url: None,
            token_url: token_endpoint,
            client_id: &config.client_id,
            scopes: Some(&scope_str),
            extra_headers: HeaderMap::new(),
            encoding: RequestEncoding::Form,
        };
        let tokens =
            poll_for_tokens(client, &df_cfg, device_code, interval_secs, expires_in_secs).await?;
        return Ok(TokenData {
            access_token: tokens.access_token,
            id_token: None,
            refresh_token: tokens.refresh_token,
            expires_at: tokens.expires_at,
        });
    }

    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(expires_in_secs);
    let mut effective_interval = interval_secs;

    let mut params = vec![
        ("client_id", config.client_id.clone()),
        ("device_code", device_code.to_string()),
        (
            "grant_type",
            "urn:ietf:params:oauth:grant-type:device_code".to_string(),
        ),
    ];
    if let Some(secret) = &config.client_secret {
        params.push(("client_secret", secret.clone()));
    }

    loop {
        if tokio::time::Instant::now() >= deadline {
            bail!("timed out waiting for user OIDC device authorization");
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(effective_interval)).await;

        let resp = client
            .post(token_endpoint)
            .form(&params)
            .send()
            .await
            .context("failed to poll OIDC token endpoint")?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .context("failed to parse OIDC device poll response")?;

        if status.is_success() && body.get("access_token").is_some() {
            return extract_token_data(&body, None, None);
        }

        match body.get("error").and_then(Value::as_str) {
            Some("authorization_pending") => continue,
            Some("slow_down") => {
                effective_interval += 5;
            }
            Some(err) => bail!("OIDC device authorization failed: {}", err),
            None => bail!(
                "Unexpected OIDC device token response ({}): {}",
                status,
                body
            ),
        }
    }
}

pub struct LiteLLMOidcAuthProvider {
    config: LiteLLMOidcConfig,
    cache: TokenCache,
    tls_config: Option<TlsConfig>,
}

impl LiteLLMOidcAuthProvider {
    pub fn new(config: LiteLLMOidcConfig, tls_config: Option<TlsConfig>) -> Self {
        let cache = TokenCache::new(&config);
        Self {
            config,
            cache,
            tls_config,
        }
    }

    #[cfg(test)]
    pub fn with_cache(
        config: LiteLLMOidcConfig,
        cache: TokenCache,
        tls_config: Option<TlsConfig>,
    ) -> Self {
        Self {
            config,
            cache,
            tls_config,
        }
    }

    pub async fn get_or_acquire_jwt(&self) -> Result<String> {
        let _guard = LITELLM_OAUTH_MUTEX.lock().await;

        if let Some(cached) = self.cache.load() {
            if !cached.is_expired() {
                if let Ok(jwt) = cached.select_jwt(self.config.token_type) {
                    return Ok(jwt);
                }
            }

            if let Some(refresh_token) = cached.refresh_token.as_deref() {
                match discover_oidc_endpoints(&self.config.issuer_url, self.tls_config.as_ref())
                    .await
                {
                    Ok(endpoints) => {
                        match refresh_oidc_token(
                            &endpoints,
                            &self.config,
                            refresh_token,
                            cached.id_token.as_deref(),
                            self.tls_config.as_ref(),
                        )
                        .await
                        {
                            Ok(refreshed) => {
                                if let Err(e) = self.cache.save(&refreshed) {
                                    tracing::warn!(
                                        "Failed to cache refreshed LiteLLM token: {}",
                                        e
                                    );
                                }
                                return refreshed.select_jwt(self.config.token_type);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "LiteLLM OIDC token refresh failed ({}), starting new login flow",
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("LiteLLM OIDC discovery failed during refresh: {}", e);
                    }
                }
            }
        }

        let token_data = self.execute_login_flow().await?;
        let jwt = token_data.select_jwt(self.config.token_type)?;
        self.cache.save(&token_data)?;
        Ok(jwt)
    }

    pub async fn force_interactive_login(&self) -> Result<()> {
        let _guard = LITELLM_OAUTH_MUTEX.lock().await;
        let previous = self.cache.load();
        self.cache.clear();

        match self.execute_login_flow().await {
            Ok(token_data) => {
                let _ = token_data.select_jwt(self.config.token_type)?;
                self.cache.save(&token_data)?;
                Ok(())
            }
            Err(e) => {
                if let Some(prev) = previous {
                    let _ = self.cache.save(&prev);
                }
                Err(e)
            }
        }
    }

    async fn execute_login_flow(&self) -> Result<TokenData> {
        let endpoints =
            discover_oidc_endpoints(&self.config.issuer_url, self.tls_config.as_ref()).await?;

        match self.config.flow {
            OidcFlowMode::Browser => {
                run_browser_pkce_flow(&endpoints, &self.config, self.tls_config.as_ref()).await
            }
            OidcFlowMode::Device => {
                run_device_code_oidc_flow(&endpoints, &self.config, self.tls_config.as_ref()).await
            }
            OidcFlowMode::Auto => {
                match run_browser_pkce_flow(&endpoints, &self.config, self.tls_config.as_ref())
                    .await
                {
                    Ok(tokens) => Ok(tokens),
                    Err(browser_err) if endpoints.device_authorization_endpoint.is_some() => {
                        tracing::warn!(
                            "LiteLLM browser OIDC flow failed ({}); falling back to device code flow",
                            browser_err
                        );
                        run_device_code_oidc_flow(
                            &endpoints,
                            &self.config,
                            self.tls_config.as_ref(),
                        )
                        .await
                    }
                    Err(browser_err) => Err(browser_err),
                }
            }
        }
    }
}

#[async_trait]
impl AuthProvider for LiteLLMOidcAuthProvider {
    async fn get_auth_header(&self) -> Result<(String, String)> {
        let jwt = self.get_or_acquire_jwt().await?;
        Ok(("Authorization".to_string(), format!("Bearer {}", jwt)))
    }

    async fn refresh_credentials(&self) -> Result<()> {
        let _guard = LITELLM_OAUTH_MUTEX.lock().await;
        if let Some(mut cached) = self.cache.load() {
            if cached.refresh_token.is_some() {
                cached.expires_at = Some(Utc::now() - Duration::seconds(120));
                self.cache.save(&cached)?;
            } else {
                self.cache.clear();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sample_config(issuer_url: String) -> LiteLLMOidcConfig {
        LiteLLMOidcConfig {
            issuer_url,
            client_id: "litellm-client".to_string(),
            client_secret: None,
            scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "offline_access".to_string(),
            ],
            audience: Some("https://litellm.example.com".to_string()),
            flow: OidcFlowMode::Device,
            redirect_url: "http://localhost".to_string(),
            token_type: OidcTokenType::AccessToken,
        }
    }

    #[tokio::test]
    async fn discovers_standard_openid_configuration() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/.well-known/openid-configuration"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "authorization_endpoint": format!("{}/authorize", server.uri()),
                "token_endpoint": format!("{}/token", server.uri()),
                "device_authorization_endpoint": format!("{}/device", server.uri())
            })))
            .mount(&server)
            .await;

        let endpoints = discover_oidc_endpoints(&server.uri(), None).await.unwrap();
        assert_eq!(
            endpoints.authorization_endpoint.as_deref(),
            Some(format!("{}/authorize", server.uri()).as_str())
        );
        assert_eq!(endpoints.token_endpoint, format!("{}/token", server.uri()));
        assert_eq!(
            endpoints.device_authorization_endpoint.as_deref(),
            Some(format!("{}/device", server.uri()).as_str())
        );
    }

    #[tokio::test]
    async fn refreshes_expired_cached_token_and_returns_bearer_jwt() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/.well-known/openid-configuration"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "token_endpoint": format!("{}/token", server.uri())
            })))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/token"))
            .and(body_string_contains("grant_type=refresh_token"))
            .and(body_string_contains("refresh_token=refresh-123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "fresh-jwt-access-token",
                "id_token": "fresh-jwt-id-token",
                "expires_in": 3600
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let mut config = sample_config(server.uri());
        config.token_type = OidcTokenType::IdToken;
        let cache = TokenCache::with_base_dir(dir.path().to_path_buf(), &config);

        cache
            .save(&TokenData {
                access_token: "expired-access".to_string(),
                id_token: Some("expired-id".to_string()),
                refresh_token: Some("refresh-123".to_string()),
                expires_at: Some(Utc::now() - Duration::seconds(10)),
            })
            .unwrap();

        let provider = LiteLLMOidcAuthProvider::with_cache(config, cache.clone(), None);
        let (header, value) = provider.get_auth_header().await.unwrap();
        assert_eq!(header, "Authorization");
        assert_eq!(value, "Bearer fresh-jwt-id-token");

        let stored = cache.load().unwrap();
        assert_eq!(stored.access_token, "fresh-jwt-access-token");
        assert_eq!(stored.id_token.as_deref(), Some("fresh-jwt-id-token"));
        assert_eq!(stored.refresh_token.as_deref(), Some("refresh-123"));
    }

    #[tokio::test]
    async fn refresh_credentials_marks_cached_token_expired_for_reactive_401_retry() {
        let dir = tempfile::tempdir().unwrap();
        let config = sample_config("http://localhost:9999".to_string());
        let cache = TokenCache::with_base_dir(dir.path().to_path_buf(), &config);

        cache
            .save(&TokenData {
                access_token: "valid-jwt".to_string(),
                id_token: None,
                refresh_token: Some("refresh-abc".to_string()),
                expires_at: Some(Utc::now() + Duration::hours(1)),
            })
            .unwrap();

        let provider = LiteLLMOidcAuthProvider::with_cache(config, cache.clone(), None);
        assert!(!cache.load().unwrap().is_expired());

        provider.refresh_credentials().await.unwrap();
        assert!(cache.load().unwrap().is_expired());
    }

    #[test]
    fn builds_authorization_url_with_pkce_and_audience() {
        let config = sample_config("https://idp.example.com".to_string());
        let url = build_authorization_url(
            "https://idp.example.com/oauth2/authorize",
            &config,
            "http://localhost:54321/callback",
            "state-xyz",
            "verifier-12345",
        )
        .unwrap();

        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=litellm-client"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("audience=https%3A%2F%2Flitellm.example.com"));
        assert!(url.contains("state=state-xyz"));
    }

    #[tokio::test]
    async fn falls_back_to_oauth_authorization_server_discovery() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/.well-known/openid-configuration"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/.well-known/oauth-authorization-server"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "authorization_endpoint": format!("{}/oauth/authorize", server.uri()),
                "token_endpoint": format!("{}/oauth/token", server.uri())
            })))
            .mount(&server)
            .await;

        let endpoints = discover_oidc_endpoints(&server.uri(), None).await.unwrap();
        assert_eq!(
            endpoints.authorization_endpoint.as_deref(),
            Some(format!("{}/oauth/authorize", server.uri()).as_str())
        );
        assert_eq!(
            endpoints.token_endpoint,
            format!("{}/oauth/token", server.uri())
        );
    }

    #[tokio::test]
    async fn rejects_non_https_remote_issuer() {
        let err = discover_oidc_endpoints("http://idp.example.com/realms/main", None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("HTTPS"));
    }

    #[tokio::test]
    async fn exchanges_authorization_code_for_jwt_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/token"))
            .and(body_string_contains("grant_type=authorization_code"))
            .and(body_string_contains("code=auth-code-789"))
            .and(body_string_contains("code_verifier=pkce-verifier-xyz"))
            .and(body_string_contains("client_secret=top-secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "exchanged-access-jwt",
                "id_token": "exchanged-id-jwt",
                "refresh_token": "exchanged-refresh-token",
                "expires_in": 1800
            })))
            .mount(&server)
            .await;

        let endpoints = OidcEndpoints {
            authorization_endpoint: Some(format!("{}/authorize", server.uri())),
            token_endpoint: format!("{}/token", server.uri()),
            device_authorization_endpoint: None,
        };
        let mut config = sample_config(server.uri());
        config.client_secret = Some("top-secret".to_string());

        let token_data = exchange_code_for_token(
            &endpoints,
            &config,
            "auth-code-789",
            "http://localhost:12345/callback",
            "pkce-verifier-xyz",
            None,
        )
        .await
        .unwrap();

        assert_eq!(token_data.access_token, "exchanged-access-jwt");
        assert_eq!(token_data.id_token.as_deref(), Some("exchanged-id-jwt"));
        assert_eq!(
            token_data.refresh_token.as_deref(),
            Some("exchanged-refresh-token")
        );
        assert!(token_data.expires_at.is_some());
    }

    #[tokio::test]
    async fn polls_device_token_handling_pending_and_extracts_id_token() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/token"))
            .and(body_string_contains("device_code=dev-code-1"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": "authorization_pending"
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/token"))
            .and(body_string_contains("device_code=dev-code-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "device-access-jwt",
                "id_token": "device-id-jwt",
                "refresh_token": "device-refresh-token",
                "expires_in": 3600
            })))
            .mount(&server)
            .await;

        let mut config = sample_config(server.uri());
        config.token_type = OidcTokenType::IdToken;
        let client = Client::new();

        let token_data = poll_oidc_device_token(
            &client,
            &format!("{}/token", server.uri()),
            &config,
            "dev-code-1",
            0,
            10,
        )
        .await
        .unwrap();

        assert_eq!(
            token_data.select_jwt(config.token_type).unwrap(),
            "device-id-jwt"
        );
    }
}
