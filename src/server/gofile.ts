const GOFILE_UPLOAD_URL = 'https://upload.gofile.io/uploadfile';

export async function uploadToGofile(params: {
  buffer: Buffer;
  fileName: string;
  mimeType?: string;
}): Promise<string> {
  const token = (process.env.GOFILE_TOKEN || '').trim();
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};

  const form = new FormData();
  const blob = new Blob([new Uint8Array(params.buffer)], {
    type: params.mimeType || 'application/octet-stream',
  });
  form.append('file', blob, params.fileName);

  const resp = await fetch(GOFILE_UPLOAD_URL, {
    method: 'POST',
    headers,
    body: form,
  });

  const payload = (await resp.json().catch(() => ({}))) as {
    status?: string;
    data?: { downloadPage?: string };
  };

  if (!resp.ok || payload.status !== 'ok' || !payload.data?.downloadPage) {
    throw new Error(`Gofile upload failed (${resp.status}): ${JSON.stringify(payload).slice(0, 180)}`);
  }

  return payload.data.downloadPage;
}
