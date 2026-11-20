import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import JSZip from 'jszip';

const AUTH_URI = 'https://accounts.google.com/o/oauth2/v2/auth';
const TOKEN_URI = 'https://oauth2.googleapis.com/token';
const USERINFO_URI = 'https://openidconnect.googleapis.com/v1/userinfo';
const DRIVE_API = 'https://www.googleapis.com/drive/v3';
const DRIVE_UPLOAD_API = 'https://www.googleapis.com/upload/drive/v3';
const DEFAULT_FOLDER_NAME = 'BearBot_Cleaned';
const STATE_TTL_MS = 10 * 60 * 1000;

export interface DriveLinkRecord {
  key: string; // "user:<id>" | "guild:<id>"
  scope: 'user' | 'guild';
  ownerId: string;
  email: string;
  refreshToken: string;
  folderId: string;
  folderUrl: string;
  clientId?: string;
  clientSecret?: string;
  linkedAt: string;
}

interface PendingState {
  scope: 'user' | 'guild';
  ownerId: string;
  redirectUri: string;
  expiresAt: number;
}

interface OAuthClientConfig {
  clientId: string;
  clientSecret: string;
  isWebClient: boolean;
  configuredRedirects: string[];
}

interface TokenCacheEntry {
  accessToken: string;
  expiresAt: number;
}

const pendingStates = new Map<string, PendingState>();
const tokenCache = new Map<string, TokenCacheEntry>();

function getDataDir(): string {
  const dir = process.env.DRIVE_DATA_DIR
    ? path.resolve(process.env.DRIVE_DATA_DIR)
    : process.env.DATA_DIR
      ? path.resolve(process.env.DATA_DIR)
      : path.resolve(process.cwd(), 'data');
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function getLinksFilePath(): string {
  return path.join(getDataDir(), 'drive_links.json');
}

function readAllLinks(): Record<string, DriveLinkRecord> {
  const fp = getLinksFilePath();
  if (!fs.existsSync(fp)) return {};
  try {
    return JSON.parse(fs.readFileSync(fp, 'utf-8')) as Record<string, DriveLinkRecord>;
  } catch {
    return {};
  }
}

function writeAllLinks(data: Record<string, DriveLinkRecord>): void {
  const fp = getLinksFilePath();
  fs.writeFileSync(fp, JSON.stringify(data, null, 2), 'utf-8');
}

export function getPublicBaseUrl(): string {
  const direct = (process.env.PUBLIC_BASE_URL || process.env.APP_URL || '').trim().replace(/\/+$/, '');
  if (direct.startsWith('http')) return direct;
  const railwayDomain = (process.env.RAILWAY_PUBLIC_DOMAIN || '').trim().replace(/\/+$/, '');
  if (railwayDomain) return `https://${railwayDomain}`;
  return '';
}

export function getOAuthClientConfig(): OAuthClientConfig | null {
  const envCid = (process.env.GOOGLE_OAUTH_CLIENT_ID || '').trim().replace(/^["']|["']$/g, '');
  const envSec = (process.env.GOOGLE_OAUTH_CLIENT_SECRET || '').trim().replace(/^["']|["']$/g, '');
  if (envCid && envSec) {
    return { clientId: envCid, clientSecret: envSec, isWebClient: true, configuredRedirects: [] };
  }

  for (const envKey of ['GOOGLE_WEB_CREDENTIALS_JSON', 'GOOGLE_CREDENTIALS_JSON']) {
    const raw = (process.env[envKey] || '').trim();
    if (!raw) continue;
    try {
      const parsed = JSON.parse(raw) as Record<string, { client_id?: string; client_secret?: string; redirect_uris?: string[] }>;
      const block = parsed.web || parsed.installed;
      if (block?.client_id && block?.client_secret) {
        return {
          clientId: block.client_id,
          clientSecret: block.client_secret,
          isWebClient: Boolean(parsed.web),
          configuredRedirects: block.redirect_uris || [],
        };
      }
    } catch {
      // ignore malformed env JSON
    }
  }

  const credPath = path.resolve(process.cwd(), 'credentials.json');
  if (fs.existsSync(credPath)) {
    try {
      const parsed = JSON.parse(fs.readFileSync(credPath, 'utf-8')) as Record<
        string,
        { client_id?: string; client_secret?: string; redirect_uris?: string[] }
      >;
      const block = parsed.web || parsed.installed;
      if (block?.client_id && block?.client_secret) {
        return {
          clientId: block.client_id,
          clientSecret: block.client_secret,
          isWebClient: Boolean(parsed.web),
          configuredRedirects: block.redirect_uris || [],
        };
      }
    } catch {
      // ignore
    }
  }
  return null;
}

export function getFallbackTokenJson(): {
  refreshToken: string;
  clientId: string;
  clientSecret: string;
  email: string;
} | null {
  const tokPath = path.resolve(process.cwd(), 'token.json');
  if (!fs.existsSync(tokPath)) return null;
  try {
    const d = JSON.parse(fs.readFileSync(tokPath, 'utf-8')) as {
      refresh_token?: string;
      client_id?: string;
      client_secret?: string;
      account?: string;
    };
    if (d.refresh_token && d.client_id && d.client_secret) {
      return {
        refreshToken: d.refresh_token,
        clientId: d.client_id,
        clientSecret: d.client_secret,
        email: d.account || 'Bot Default Drive (token.json)',
      };
    }
  } catch {
    // ignore
  }
  return null;
}

export function getRedirectUri(cfg: OAuthClientConfig): { uri: string; isLocalhostManual: boolean } {
  const base = getPublicBaseUrl();
  if (cfg.isWebClient && base) {
    return { uri: `${base}/oauth/callback`, isLocalhostManual: false };
  }
  const hasWebCallback = cfg.configuredRedirects.some((u) => u.includes('/oauth/callback'));
  if (hasWebCallback && base) {
    return { uri: `${base}/oauth/callback`, isLocalhostManual: false };
  }
  return { uri: cfg.configuredRedirects[0] || 'http://localhost', isLocalhostManual: true };
}

export function createAuthUrl(
  scope: 'user' | 'guild',
  ownerId: string
): { url: string; state: string; isLocalhostManual: boolean } | { error: string } {
  const cfg = getOAuthClientConfig();
  if (!cfg) {
    return { error: 'Google OAuth credentials not found (credentials.json or GOOGLE_OAUTH_CLIENT_ID/SECRET).' };
  }

  const { uri, isLocalhostManual } = getRedirectUri(cfg);
  const state = crypto.randomBytes(18).toString('hex');
  pendingStates.set(state, {
    scope,
    ownerId,
    redirectUri: uri,
    expiresAt: Date.now() + STATE_TTL_MS,
  });

  // Prune expired states
  const now = Date.now();
  for (const [k, v] of pendingStates.entries()) {
    if (v.expiresAt < now) pendingStates.delete(k);
  }

  const scopes =
    (process.env.GOOGLE_DRIVE_SCOPE || '').trim() ||
    'https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/userinfo.email';

  const params = new URLSearchParams({
    client_id: cfg.clientId,
    redirect_uri: uri,
    response_type: 'code',
    scope: scopes,
    access_type: 'offline',
    prompt: 'consent',
    state,
  });

  return {
    url: `${AUTH_URI}?${params.toString()}`,
    state,
    isLocalhostManual,
  };
}

async function fetchUserEmail(accessToken: string): Promise<string> {
  try {
    const r = await fetch(USERINFO_URI, {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (r.ok) {
      const data = (await r.json()) as { email?: string };
      if (data.email) return data.email;
    }
  } catch {
    // ignore
  }
  return 'Google Drive User';
}

export async function ensureDriveFolder(
  accessToken: string,
  folderName = DEFAULT_FOLDER_NAME
): Promise<{ folderId: string; folderUrl: string }> {
  const q = encodeURIComponent(
    `name='${folderName.replace(/'/g, "\\'")}' and mimeType='application/vnd.google-apps.folder' and trashed=false`
  );
  const listRes = await fetch(`${DRIVE_API}/files?q=${q}&fields=files(id,name)&pageSize=1`, {
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  if (listRes.ok) {
    const data = (await listRes.json()) as { files?: { id: string }[] };
    if (data.files && data.files.length > 0) {
      const fid = data.files[0].id;
      return { folderId: fid, folderUrl: `https://drive.google.com/drive/folders/${fid}` };
    }
  }

  const createRes = await fetch(`${DRIVE_API}/files?fields=id`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      name: folderName,
      mimeType: 'application/vnd.google-apps.folder',
    }),
  });
  if (!createRes.ok) {
    throw new Error(`Failed to create Drive folder (${createRes.status})`);
  }
  const created = (await createRes.json()) as { id: string };
  return {
    folderId: created.id,
    folderUrl: `https://drive.google.com/drive/folders/${created.id}`,
  };
}

export async function completeOAuthCodeExchange(params: {
  code: string;
  scope: 'user' | 'guild';
  ownerId: string;
  redirectUri: string;
}): Promise<DriveLinkRecord> {
  const cfg = getOAuthClientConfig();
  if (!cfg) {
    throw new Error('Google OAuth client config missing');
  }

  const body = new URLSearchParams({
    code: params.code.trim(),
    client_id: cfg.clientId,
    client_secret: cfg.clientSecret,
    redirect_uri: params.redirectUri,
    grant_type: 'authorization_code',
  });

  const tokRes = await fetch(TOKEN_URI, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  });
  if (!tokRes.ok) {
    const errTxt = await tokRes.text().catch(() => '');
    throw new Error(`Google token exchange failed (${tokRes.status}): ${errTxt.slice(0, 200)}`);
  }

  const tokJson = (await tokRes.json()) as {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
  };
  if (!tokJson.access_token) {
    throw new Error('Google did not return an access_token');
  }

  const key = `${params.scope}:${params.ownerId}`;
  const existing = readAllLinks()[key];
  const refreshToken = tokJson.refresh_token || existing?.refreshToken;
  if (!refreshToken) {
    throw new Error('Google did not return a refresh_token — please try linking again.');
  }

  const email = await fetchUserEmail(tokJson.access_token);
  const { folderId, folderUrl } = await ensureDriveFolder(tokJson.access_token);

  const record: DriveLinkRecord = {
    key,
    scope: params.scope,
    ownerId: params.ownerId,
    email,
    refreshToken,
    folderId,
    folderUrl,
    clientId: cfg.clientId,
    clientSecret: cfg.clientSecret,
    linkedAt: new Date().toISOString(),
  };

  const all = readAllLinks();
  all[key] = record;
  writeAllLinks(all);

  tokenCache.set(key, {
    accessToken: tokJson.access_token,
    expiresAt: Date.now() + ((tokJson.expires_in || 3600) - 120) * 1000,
  });

  return record;
}

export async function handleOAuthCallback(
  code: string,
  state: string
): Promise<{ ok: boolean; message: string; record?: DriveLinkRecord }> {
  const pending = pendingStates.get(state);
  if (!pending || pending.expiresAt < Date.now()) {
    pendingStates.delete(state);
    return { ok: false, message: 'الرابط منتهي أو غير صالح — اطلب رابط جديد من /profile.' };
  }
  pendingStates.delete(state);

  try {
    const record = await completeOAuthCodeExchange({
      code,
      scope: pending.scope,
      ownerId: pending.ownerId,
      redirectUri: pending.redirectUri,
    });
    return {
      ok: true,
      message: `تم ربط حساب Google Drive بنجاح (${record.email})! يمكنك إغلاق هذه الصفحة والعودة إلى Discord.`,
      record,
    };
  } catch (err) {
    return { ok: false, message: `فشل ربط Google Drive: ${(err as Error).message}` };
  }
}

export async function completeManualCodeFromUser(
  scope: 'user' | 'guild',
  ownerId: string,
  pastedUrlOrCode: string
): Promise<DriveLinkRecord> {
  const raw = pastedUrlOrCode.trim();
  let code = raw;
  if (raw.startsWith('http')) {
    const u = new URL(raw);
    const extracted = u.searchParams.get('code');
    if (!extracted) throw new Error('الرابط لا يحتوي على ?code=');
    code = extracted;
  }
  const cfg = getOAuthClientConfig();
  if (!cfg) throw new Error('Google OAuth config missing');
  const { uri } = getRedirectUri(cfg);
  return completeOAuthCodeExchange({ code, scope, ownerId, redirectUri: uri });
}

export function getDriveLink(scope: 'user' | 'guild', ownerId: string): DriveLinkRecord | null {
  return readAllLinks()[`${scope}:${ownerId}`] || null;
}

export function unlinkDrive(scope: 'user' | 'guild', ownerId: string): boolean {
  const key = `${scope}:${ownerId}`;
  const all = readAllLinks();
  if (!all[key]) return false;
  delete all[key];
  writeAllLinks(all);
  tokenCache.delete(key);
  return true;
}

export function updateDriveFolder(
  scope: 'user' | 'guild',
  ownerId: string,
  folderLinkOrId: string
): DriveLinkRecord | null {
  const key = `${scope}:${ownerId}`;
  const all = readAllLinks();
  const rec = all[key];
  if (!rec) return null;
  const parsed = parseDriveUrl(folderLinkOrId);
  const folderId = parsed?.id || folderLinkOrId.trim();
  rec.folderId = folderId;
  rec.folderUrl = `https://drive.google.com/drive/folders/${folderId}`;
  all[key] = rec;
  writeAllLinks(all);
  return rec;
}

async function refreshAccessToken(
  cacheKey: string,
  refreshToken: string,
  clientId: string,
  clientSecret: string
): Promise<string> {
  const cached = tokenCache.get(cacheKey);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.accessToken;
  }

  const body = new URLSearchParams({
    client_id: clientId,
    client_secret: clientSecret,
    refresh_token: refreshToken,
    grant_type: 'refresh_token',
  });
  const res = await fetch(TOKEN_URI, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`Drive token refresh failed (${res.status}): ${errText.slice(0, 160)}`);
  }
  const data = (await res.json()) as { access_token: string; expires_in?: number };
  tokenCache.set(cacheKey, {
    accessToken: data.access_token,
    expiresAt: Date.now() + ((data.expires_in || 3600) - 120) * 1000,
  });
  return data.access_token;
}

export async function resolveDriveAuth(
  userId: string,
  guildId?: string | null
): Promise<{
  accessToken: string;
  folderId: string;
  folderUrl: string;
  email: string;
  sourceLabel: string;
} | null> {
  const userRec = getDriveLink('user', userId);
  const guildRec = guildId ? getDriveLink('guild', guildId) : null;
  const cfg = getOAuthClientConfig();

  for (const [rec, label] of [
    [userRec, '👤 درايفك الشخصي'],
    [guildRec, '🏠 درايف السيرفر'],
  ] as const) {
    if (!rec) continue;
    const cid = rec.clientId || cfg?.clientId;
    const csec = rec.clientSecret || cfg?.clientSecret;
    if (!cid || !csec) continue;
    try {
      const accessToken = await refreshAccessToken(rec.key, rec.refreshToken, cid, csec);
      return {
        accessToken,
        folderId: rec.folderId,
        folderUrl: rec.folderUrl,
        email: rec.email,
        sourceLabel: label,
      };
    } catch (err) {
      console.warn(`Drive auth refresh failed for ${rec.key}:`, (err as Error).message);
    }
  }

  // Fallback to token.json imported from TBATE_Bot (safe try/catch so expired token never breaks /clean)
  const fallback = getFallbackTokenJson();
  if (fallback) {
    try {
      const accessToken = await refreshAccessToken(
        'bot:default',
        fallback.refreshToken,
        fallback.clientId,
        fallback.clientSecret
      );
      const { folderId, folderUrl } = await ensureDriveFolder(accessToken, DEFAULT_FOLDER_NAME);
      return {
        accessToken,
        folderId,
        folderUrl,
        email: fallback.email,
        sourceLabel: '🤖 درايف البوت الافتراضي',
      };
    } catch (err) {
      console.warn('Fallback token.json refresh failed:', (err as Error).message);
    }
  }

  return null;
}

export function parseDriveUrl(input: string): { type: 'folder' | 'file'; id: string } | null {
  const s = input.trim();
  const folderMatch = s.match(/\/folders\/([a-zA-Z0-9_-]{10,})/);
  if (folderMatch) return { type: 'folder', id: folderMatch[1] };
  const fileMatch = s.match(/\/file\/d\/([a-zA-Z0-9_-]{10,})/);
  if (fileMatch) return { type: 'file', id: fileMatch[1] };
  const idMatch = s.match(/[?&]id=([a-zA-Z0-9_-]{10,})/);
  if (idMatch) return { type: 'file', id: idMatch[1] };
  if (/^[a-zA-Z0-9_-]{20,}$/.test(s)) return { type: 'file', id: s };
  return null;
}

function naturalSortNames(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

export async function downloadFromDriveLink(
  driveUrl: string,
  accessToken: string
): Promise<{ fileName: string; buffer: Buffer; sourceFolderId?: string }> {
  const parsed = parseDriveUrl(driveUrl);
  if (!parsed) {
    throw new Error('رابط Google Drive غير صالح — ابعت رابط ملف أو فولدر من drive.google.com');
  }

  const headers = { Authorization: `Bearer ${accessToken}` };

  // Fetch metadata to verify whether it's a folder or file
  const metaRes = await fetch(
    `${DRIVE_API}/files/${encodeURIComponent(parsed.id)}?fields=id,name,mimeType,parents&supportsAllDrives=true`,
    { headers }
  );
  if (!metaRes.ok) {
    throw new Error(
      `تعذّر الوصول لرابط Google Drive (${metaRes.status}) — تأكد إن الملف/الفولدر متشارك أو تابع لحسابك المربوط في /profile.`
    );
  }
  const meta = (await metaRes.json()) as {
    id: string;
    name: string;
    mimeType: string;
    parents?: string[];
  };

  if (meta.mimeType !== 'application/vnd.google-apps.folder') {
    const dlRes = await fetch(
      `${DRIVE_API}/files/${encodeURIComponent(meta.id)}?alt=media&supportsAllDrives=true`,
      { headers }
    );
    if (!dlRes.ok) {
      throw new Error(`فشل تحميل الملف من Google Drive (${dlRes.status})`);
    }
    const buf = Buffer.from(await dlRes.arrayBuffer());
    return {
      fileName: meta.name || 'drive_file.zip',
      buffer: buf,
      sourceFolderId: meta.parents?.[0],
    };
  }

  // Folder: list images or ZIP inside folder
  const q = encodeURIComponent(`'${meta.id}' in parents and trashed=false`);
  const listRes = await fetch(
    `${DRIVE_API}/files?q=${q}&fields=files(id,name,mimeType)&pageSize=200&supportsAllDrives=true&includeItemsFromAllDrives=true`,
    { headers }
  );
  if (!listRes.ok) {
    throw new Error(`فشل قراءة محتويات فولدر Google Drive (${listRes.status})`);
  }
  const listed = (await listRes.json()) as {
    files?: { id: string; name: string; mimeType: string }[];
  };
  const files = listed.files || [];

  const zipFiles = files.filter((f) => /\.zip$/i.test(f.name));
  const imgFiles = files
    .filter((f) => /\.(png|jpe?g|webp)$/i.test(f.name))
    .sort((a, b) => naturalSortNames(a.name, b.name));

  if (imgFiles.length === 0 && zipFiles.length === 1) {
    const zf = zipFiles[0];
    const dlRes = await fetch(
      `${DRIVE_API}/files/${encodeURIComponent(zf.id)}?alt=media&supportsAllDrives=true`,
      { headers }
    );
    if (!dlRes.ok) throw new Error(`فشل تحميل الـ ZIP من الفولدر (${dlRes.status})`);
    return {
      fileName: zf.name,
      buffer: Buffer.from(await dlRes.arrayBuffer()),
      sourceFolderId: meta.id,
    };
  }

  if (imgFiles.length === 0) {
    throw new Error('فولدر Google Drive لا يحتوي على صور (PNG/JPG/WebP) أو ملف ZIP.');
  }

  if (imgFiles.length === 1) {
    const img = imgFiles[0];
    const dlRes = await fetch(
      `${DRIVE_API}/files/${encodeURIComponent(img.id)}?alt=media&supportsAllDrives=true`,
      { headers }
    );
    if (!dlRes.ok) throw new Error(`فشل تحميل الصورة من الفولدر (${dlRes.status})`);
    return {
      fileName: img.name,
      buffer: Buffer.from(await dlRes.arrayBuffer()),
      sourceFolderId: meta.id,
    };
  }

  // Pack all images in folder into a ZIP for the cleaning pipeline
  const zip = new JSZip();
  for (const img of imgFiles) {
    const dlRes = await fetch(
      `${DRIVE_API}/files/${encodeURIComponent(img.id)}?alt=media&supportsAllDrives=true`,
      { headers }
    );
    if (!dlRes.ok) continue;
    const ab = await dlRes.arrayBuffer();
    zip.file(img.name, Buffer.from(ab));
  }
  const packed = await zip.generateAsync({ type: 'nodebuffer', compression: 'STORE' });
  const safeName = (meta.name || 'drive_chapter').replace(/[^\w\-. ]+/g, '_');
  return {
    fileName: `${safeName}.zip`,
    buffer: packed,
    sourceFolderId: meta.id,
  };
}

export async function uploadToDrive(params: {
  accessToken: string;
  parentFolderId: string;
  fileName: string;
  buffer: Buffer;
  mimeType: string;
}): Promise<{ fileId: string; webViewLink: string }> {
  const boundary = `bearbot_${crypto.randomBytes(12).toString('hex')}`;
  const metadata = JSON.stringify({
    name: params.fileName,
    parents: [params.parentFolderId],
  });

  const head = Buffer.from(
    `--${boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n${metadata}\r\n` +
      `--${boundary}\r\nContent-Type: ${params.mimeType}\r\n\r\n`,
    'utf-8'
  );
  const tail = Buffer.from(`\r\n--${boundary}--`, 'utf-8');
  const body = Buffer.concat([head, params.buffer, tail]);

  const upRes = await fetch(
    `${DRIVE_UPLOAD_API}/files?uploadType=multipart&fields=id,name,webViewLink&supportsAllDrives=true`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${params.accessToken}`,
        'Content-Type': `multipart/related; boundary=${boundary}`,
        'Content-Length': String(body.byteLength),
      },
      body,
    }
  );
  if (!upRes.ok) {
    const errTxt = await upRes.text().catch(() => '');
    throw new Error(`فشل رفع النتيجة على Google Drive (${upRes.status}): ${errTxt.slice(0, 180)}`);
  }

  const uploaded = (await upRes.json()) as { id: string; webViewLink?: string };

  // Make file viewable by anyone with the link (best-effort)
  await fetch(
    `${DRIVE_API}/files/${encodeURIComponent(uploaded.id)}/permissions?supportsAllDrives=true`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${params.accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ role: 'reader', type: 'anyone' }),
    }
  ).catch(() => {});

  return {
    fileId: uploaded.id,
    webViewLink: uploaded.webViewLink || `https://drive.google.com/file/d/${uploaded.id}/view`,
  };
}
