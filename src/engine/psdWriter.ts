/**
 * Zero-dependency Adobe Photoshop (.psd) writer (PSD v1, 8-bit RGB/RGBA, PackBits RLE).
 * Produces a 2-layer editable PSD:
 *   - Layer 1 (bottom): "Original" — untouched original scan
 *   - Layer 2 (top):    "Cleaned"  — only modified/cleaned pixels (transparent elsewhere)
 *   - Composite image:  flattened cleaned page for instant thumbnails & previews
 */

function packBitsEncodeRow(row: Uint8Array): Uint8Array {
  const len = row.length;
  const out: number[] = [];
  let i = 0;
  while (i < len) {
    // Check for run of >= 3 identical bytes
    let runLen = 1;
    while (i + runLen < len && runLen < 128 && row[i + runLen] === row[i]) {
      runLen++;
    }
    if (runLen >= 3) {
      out.push((257 - runLen) & 0xff, row[i]);
      i += runLen;
    } else {
      const litStart = i;
      let litLen = 0;
      while (i < len && litLen < 128) {
        if (i + 2 < len && row[i] === row[i + 1] && row[i] === row[i + 2]) {
          break;
        }
        i++;
        litLen++;
      }
      out.push(litLen - 1);
      for (let k = litStart; k < litStart + litLen; k++) {
        out.push(row[k]);
      }
    }
  }
  return Uint8Array.from(out);
}

function encodeChannelRle(channel: Uint8Array, width: number, height: number): Buffer {
  const rowBuffers: Uint8Array[] = new Array(height);
  const rowCounts = Buffer.alloc(height * 2);
  let totalDataBytes = 0;
  for (let y = 0; y < height; y++) {
    const row = channel.subarray(y * width, (y + 1) * width);
    const encoded = packBitsEncodeRow(row);
    rowBuffers[y] = encoded;
    rowCounts.writeUInt16BE(encoded.length, y * 2);
    totalDataBytes += encoded.length;
  }
  const result = Buffer.alloc(2 + height * 2 + totalDataBytes);
  result.writeUInt16BE(1, 0); // Compression = 1 (RLE / PackBits)
  rowCounts.copy(result, 2);
  let offset = 2 + height * 2;
  for (let y = 0; y < height; y++) {
    Buffer.from(rowBuffers[y].buffer, rowBuffers[y].byteOffset, rowBuffers[y].byteLength).copy(
      result,
      offset
    );
    offset += rowBuffers[y].length;
  }
  return result;
}

interface PsdLayerInput {
  name: string;
  rgba: Uint8Array;
  width: number;
  height: number;
}

function buildLayerSection(layers: PsdLayerInput[]): Buffer {
  const layerRecords: Buffer[] = [];
  const layerChannelData: Buffer[] = [];

  for (const layer of layers) {
    const { width, height, rgba, name } = layer;
    const pixelCount = width * height;
    const chA = new Uint8Array(pixelCount);
    const chR = new Uint8Array(pixelCount);
    const chG = new Uint8Array(pixelCount);
    const chB = new Uint8Array(pixelCount);

    for (let i = 0, p = 0; i < pixelCount; i++, p += 4) {
      chR[i] = rgba[p];
      chG[i] = rgba[p + 1];
      chB[i] = rgba[p + 2];
      chA[i] = rgba[p + 3];
    }

    // Channel IDs: -1 (Alpha), 0 (Red), 1 (Green), 2 (Blue)
    const encodedChannels: { id: number; data: Buffer }[] = [
      { id: -1, data: encodeChannelRle(chA, width, height) },
      { id: 0, data: encodeChannelRle(chR, width, height) },
      { id: 1, data: encodeChannelRle(chG, width, height) },
      { id: 2, data: encodeChannelRle(chB, width, height) },
    ];

    const nameBytes = Buffer.from(name.slice(0, 255), 'ascii');
    const pascalLen = 1 + nameBytes.length;
    const paddedPascalLen = Math.ceil(pascalLen / 4) * 4;
    const pascalBuf = Buffer.alloc(paddedPascalLen);
    pascalBuf.writeUInt8(nameBytes.length, 0);
    nameBytes.copy(pascalBuf, 1);

    // Extra data: mask(4) + blending ranges(4) + pascal name
    const extraLen = 4 + 4 + paddedPascalLen;
    const record = Buffer.alloc(16 + 2 + encodedChannels.length * 6 + 4 + 4 + 4 + 4 + extraLen);
    let off = 0;
    record.writeInt32BE(0, off); off += 4; // top
    record.writeInt32BE(0, off); off += 4; // left
    record.writeInt32BE(height, off); off += 4; // bottom
    record.writeInt32BE(width, off); off += 4; // right
    record.writeUInt16BE(encodedChannels.length, off); off += 2;

    for (const ch of encodedChannels) {
      record.writeInt16BE(ch.id, off); off += 2;
      record.writeUInt32BE(ch.data.length, off); off += 4;
      layerChannelData.push(ch.data);
    }

    record.write('8BIM', off, 'ascii'); off += 4;
    record.write('norm', off, 'ascii'); off += 4;
    record.writeUInt8(255, off); off += 1; // opacity
    record.writeUInt8(0, off); off += 1; // clipping
    record.writeUInt8(0, off); off += 1; // flags (visible)
    record.writeUInt8(0, off); off += 1; // filler
    record.writeUInt32BE(extraLen, off); off += 4;
    record.writeUInt32BE(0, off); off += 4; // layer mask length = 0
    record.writeUInt32BE(0, off); off += 4; // blending ranges length = 0
    pascalBuf.copy(record, off);

    layerRecords.push(record);
  }

  const layerCountBuf = Buffer.alloc(2);
  layerCountBuf.writeInt16BE(layers.length, 0);

  let layerInfoBody = Buffer.concat([layerCountBuf, ...layerRecords, ...layerChannelData]);
  if (layerInfoBody.length % 2 !== 0) {
    layerInfoBody = Buffer.concat([layerInfoBody, Buffer.alloc(1)]);
  }

  const layerInfoLen = Buffer.alloc(4);
  layerInfoLen.writeUInt32BE(layerInfoBody.length, 0);
  const globalMaskLen = Buffer.alloc(4); // 0 bytes

  const sectionBody = Buffer.concat([layerInfoLen, layerInfoBody, globalMaskLen]);
  const sectionLen = Buffer.alloc(4);
  sectionLen.writeUInt32BE(sectionBody.length, 0);
  return Buffer.concat([sectionLen, sectionBody]);
}

function buildCompositeImageData(rgba: Uint8Array, width: number, height: number): Buffer {
  const pixelCount = width * height;
  const planes = [
    new Uint8Array(pixelCount),
    new Uint8Array(pixelCount),
    new Uint8Array(pixelCount),
  ];
  for (let i = 0, p = 0; i < pixelCount; i++, p += 4) {
    planes[0][i] = rgba[p];
    planes[1][i] = rgba[p + 1];
    planes[2][i] = rgba[p + 2];
  }

  const totalRows = height * 3;
  const rowCounts = Buffer.alloc(totalRows * 2);
  const encodedRows: Uint8Array[] = new Array(totalRows);
  let totalBytes = 0;
  let rIdx = 0;

  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < height; y++) {
      const enc = packBitsEncodeRow(planes[c].subarray(y * width, (y + 1) * width));
      encodedRows[rIdx] = enc;
      rowCounts.writeUInt16BE(enc.length, rIdx * 2);
      totalBytes += enc.length;
      rIdx++;
    }
  }

  const out = Buffer.alloc(2 + totalRows * 2 + totalBytes);
  out.writeUInt16BE(1, 0); // RLE PackBits
  rowCounts.copy(out, 2);
  let off = 2 + totalRows * 2;
  for (let i = 0; i < totalRows; i++) {
    Buffer.from(encodedRows[i].buffer, encodedRows[i].byteOffset, encodedRows[i].byteLength).copy(
      out,
      off
    );
    off += encodedRows[i].length;
  }
  return out;
}

export function createTwoLayerPsd(params: {
  width: number;
  height: number;
  originalRgba: Uint8Array;
  cleanedRgba: Uint8Array;
}): Buffer {
  const { width, height, originalRgba, cleanedRgba } = params;
  const pixelCount = width * height;

  // Build sparse diff overlay for top "Cleaned" layer (transparent where unchanged!)
  const diffOverlay = new Uint8Array(pixelCount * 4);
  for (let i = 0; i < pixelCount * 4; i += 4) {
    if (
      originalRgba[i] !== cleanedRgba[i] ||
      originalRgba[i + 1] !== cleanedRgba[i + 1] ||
      originalRgba[i + 2] !== cleanedRgba[i + 2]
    ) {
      diffOverlay[i] = cleanedRgba[i];
      diffOverlay[i + 1] = cleanedRgba[i + 1];
      diffOverlay[i + 2] = cleanedRgba[i + 2];
      diffOverlay[i + 3] = 255;
    }
  }

  // 26-byte PSD Header (3 channels RGB composite, 8-bit)
  const header = Buffer.alloc(26);
  header.write('8BPS', 0, 'ascii');
  header.writeUInt16BE(1, 4); // Version 1 (PSD)
  header.writeUInt16BE(3, 12); // Channels (RGB)
  header.writeUInt32BE(height, 14);
  header.writeUInt32BE(width, 18);
  header.writeUInt16BE(8, 22); // Depth = 8 bits
  header.writeUInt16BE(3, 24); // Color Mode = RGB (3)

  const colorModeLen = Buffer.alloc(4); // 0
  const imageResourcesLen = Buffer.alloc(4); // 0
  const layerSection = buildLayerSection([
    { name: 'Original', rgba: originalRgba, width, height },
    { name: 'Cleaned', rgba: diffOverlay, width, height },
  ]);
  const composite = buildCompositeImageData(cleanedRgba, width, height);

  return Buffer.concat([header, colorModeLen, imageResourcesLen, layerSection, composite]);
}
