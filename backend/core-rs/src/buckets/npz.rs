//! Minimal, defensive reader for numpy `.npz` (zip, STORED members only, as
//! written by `np.savez`) and `.npy` files.
//!
//! Only what the abstraction assets use: little-endian integer / float64 /
//! unicode dtypes, C order (2-D Fortran order is transposed). Everything read
//! from the file is bounds-checked against the file size before allocating, so a
//! malformed or hostile file yields an `Err`, never a panic or an unbounded
//! allocation. Compressed (`np.savez_compressed`) archives are rejected with a
//! clear message (the baked tables and centroids are written uncompressed).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAX_NPY_HEADER: usize = 64 * 1024;
const MAX_CENTRAL_DIR: u64 = 16 * 1024 * 1024;
const MAX_MEMBERS: usize = 4096;
const CHUNK: usize = 1 << 20;

/// Decoded array payload.
#[derive(Debug, Clone, PartialEq)]
pub enum NpyData {
    /// `|u1` / `<u1`
    U8(Vec<u8>),
    /// any other little-endian integer dtype, widened
    I64(Vec<i64>),
    /// `<f8` (or `<f4`, widened exactly)
    F64(Vec<f64>),
    /// `<U..` unicode strings (one per element)
    Str(Vec<String>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The single integer of a 0-d / 1-element integer array.
    pub fn scalar_i64(&self) -> Result<i64, String> {
        match &self.data {
            NpyData::I64(v) if v.len() == 1 => Ok(v[0]),
            NpyData::U8(v) if v.len() == 1 => Ok(v[0] as i64),
            _ => Err(format!(
                "expected an integer scalar, got shape {:?}",
                self.shape
            )),
        }
    }

    pub fn scalar_str(&self) -> Result<String, String> {
        match &self.data {
            NpyData::Str(v) if v.len() == 1 => Ok(v[0].clone()),
            _ => Err(format!(
                "expected a string scalar, got shape {:?}",
                self.shape
            )),
        }
    }
}

fn le_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn le_u64(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    U8,
    Int { size: usize, signed: bool },
    F64,
    F32,
    Unicode { chars: usize },
}

impl Kind {
    fn item_size(self) -> usize {
        match self {
            Kind::U8 => 1,
            Kind::Int { size, .. } => size,
            Kind::F64 => 8,
            Kind::F32 => 4,
            Kind::Unicode { chars } => 4 * chars,
        }
    }
}

fn parse_descr(d: &str) -> Result<Kind, String> {
    let k = match d {
        "|u1" | "<u1" => Kind::U8,
        "|i1" | "<i1" => Kind::Int {
            size: 1,
            signed: true,
        },
        "<i2" => Kind::Int {
            size: 2,
            signed: true,
        },
        "<u2" => Kind::Int {
            size: 2,
            signed: false,
        },
        "<i4" => Kind::Int {
            size: 4,
            signed: true,
        },
        "<u4" => Kind::Int {
            size: 4,
            signed: false,
        },
        "<i8" => Kind::Int {
            size: 8,
            signed: true,
        },
        "<f8" => Kind::F64,
        "<f4" => Kind::F32,
        _ => {
            if let Some(n) = d.strip_prefix("<U") {
                let chars: usize = n.parse().map_err(|_| format!("bad unicode dtype {d:?}"))?;
                if chars == 0 || chars > 4096 {
                    return Err(format!("unsupported unicode width {d:?}"));
                }
                Kind::Unicode { chars }
            } else {
                return Err(format!("unsupported npy dtype {d:?}"));
            }
        }
    };
    Ok(k)
}

/// Value of `'key': <value>` in a numpy header dict (value up to the next
/// top-level `,` or `}`; parentheses are kept together).
fn header_field<'a>(h: &'a str, key: &str) -> Result<&'a str, String> {
    let pat = format!("'{key}':");
    let start = h
        .find(&pat)
        .ok_or_else(|| format!("npy header lacks {key:?}"))?
        + pat.len();
    let rest = h[start..].trim_start();
    let mut depth = 0i32;
    for (i, ch) in rest.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' | '}' if depth == 0 => return Ok(rest[..i].trim()),
            _ => {}
        }
        if depth < 0 {
            return Err("npy header: unbalanced parentheses".into());
        }
    }
    Err(format!("npy header: unterminated {key:?}"))
}

struct Header {
    kind: Kind,
    fortran: bool,
    shape: Vec<usize>,
    /// bytes from the start of the npy blob to the first data byte
    data_start: usize,
}

fn parse_header(pre: &[u8]) -> Result<Header, String> {
    if pre.len() < 10 || &pre[..6] != b"\x93NUMPY" {
        return Err("not an npy blob (bad magic)".into());
    }
    let (hlen, hstart) = match pre[6] {
        1 => (le_u16(pre, 8) as usize, 10),
        2 | 3 => {
            if pre.len() < 12 {
                return Err("truncated npy header".into());
            }
            (le_u32(pre, 8) as usize, 12)
        }
        v => return Err(format!("unsupported npy version {v}")),
    };
    if hlen > MAX_NPY_HEADER {
        return Err(format!("npy header too large ({hlen} bytes)"));
    }
    if pre.len() < hstart + hlen {
        return Err("truncated npy header".into());
    }
    let h = std::str::from_utf8(&pre[hstart..hstart + hlen])
        .map_err(|_| "npy header is not UTF-8".to_string())?;
    let descr = header_field(h, "descr")?;
    let descr = descr
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
        .ok_or_else(|| format!("unsupported npy descr {descr}"))?;
    let kind = parse_descr(descr)?;
    let fortran = match header_field(h, "fortran_order")? {
        "False" => false,
        "True" => true,
        v => return Err(format!("bad fortran_order {v}")),
    };
    let sh = header_field(h, "shape")?;
    let inner = sh
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| format!("bad npy shape {sh}"))?;
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        shape.push(
            p.parse::<usize>()
                .map_err(|_| format!("bad npy shape {sh}"))?,
        );
        if shape.len() > 8 {
            return Err("npy array has too many dimensions".into());
        }
    }
    Ok(Header {
        kind,
        fortran,
        shape,
        data_start: hstart + hlen,
    })
}

/// Read one npy blob of exactly `size` bytes starting at the reader's position.
fn read_npy<R: Read>(r: &mut R, size: u64) -> Result<NpyArray, String> {
    let io = |e: std::io::Error| format!("read error: {e}");
    // Read the fixed preamble, then the rest of the header.
    let mut pre = vec![0u8; 12.min(size as usize)];
    r.read_exact(&mut pre).map_err(io)?;
    let hlen = if pre.len() >= 10 && pre[6] == 1 {
        le_u16(&pre, 8) as usize + 10
    } else if pre.len() >= 12 {
        le_u32(&pre, 8) as usize + 12
    } else {
        return Err("truncated npy blob".into());
    };
    if hlen > MAX_NPY_HEADER + 12 || hlen as u64 > size {
        return Err(format!(
            "npy header length {hlen} invalid for a {size}-byte member"
        ));
    }
    if hlen > pre.len() {
        let mut more = vec![0u8; hlen - pre.len()];
        r.read_exact(&mut more).map_err(io)?;
        pre.extend_from_slice(&more);
    }
    let h = parse_header(&pre)?;
    let count = h
        .shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or("npy shape overflows")?;
    let item = h.kind.item_size();
    let nbytes = count.checked_mul(item).ok_or("npy size overflows")?;
    // pre may already hold data bytes beyond the header (when hlen < 12)
    let consumed = pre.len();
    if (h.data_start as u64).checked_add(nbytes as u64) != Some(size) {
        return Err(format!(
            "npy payload size mismatch: header says {nbytes} data bytes, member holds {}",
            size.saturating_sub(h.data_start as u64)
        ));
    }
    let mut leftover: Vec<u8> = pre[h.data_start.min(consumed)..].to_vec();
    // Stream the payload in chunks, decoding as we go (bounded by `size`).
    let mut remaining = nbytes - leftover.len().min(nbytes);
    let mut raw: Vec<u8> = Vec::new();
    let mut out_u8: Vec<u8> = Vec::new();
    let mut out_i: Vec<i64> = Vec::new();
    let mut out_f: Vec<f64> = Vec::new();
    match h.kind {
        Kind::U8 => out_u8.reserve_exact(count),
        Kind::Int { .. } => out_i.reserve_exact(count),
        Kind::F64 | Kind::F32 => out_f.reserve_exact(count),
        Kind::Unicode { .. } => raw.reserve_exact(nbytes),
    }
    let mut buf = vec![0u8; CHUNK.max(item) / item * item];
    loop {
        // decode whole items from `leftover`
        let whole = leftover.len() / item * item;
        let chunk = &leftover[..whole];
        match h.kind {
            Kind::U8 => out_u8.extend_from_slice(chunk),
            Kind::Int { size, signed } => {
                for c in chunk.chunks_exact(size) {
                    let mut a = [0u8; 8];
                    a[..size].copy_from_slice(c);
                    if signed && c[size - 1] & 0x80 != 0 {
                        a[size..].fill(0xFF);
                    }
                    out_i.push(i64::from_le_bytes(a));
                }
            }
            Kind::F64 => {
                out_f.extend(
                    chunk
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|c| f64::from_le_bytes(*c)),
                );
            }
            Kind::F32 => {
                out_f.extend(
                    chunk
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c) as f64),
                );
            }
            Kind::Unicode { .. } => raw.extend_from_slice(chunk),
        }
        leftover.drain(..whole);
        if remaining == 0 {
            break;
        }
        let n = remaining.min(buf.len());
        r.read_exact(&mut buf[..n]).map_err(io)?;
        leftover.extend_from_slice(&buf[..n]);
        remaining -= n;
    }
    let data = match h.kind {
        Kind::U8 => NpyData::U8(out_u8),
        Kind::Int { .. } => NpyData::I64(out_i),
        Kind::F64 | Kind::F32 => NpyData::F64(out_f),
        Kind::Unicode { chars } => {
            let mut v = Vec::with_capacity(count);
            for c in raw.chunks_exact(4 * chars) {
                let mut s = String::new();
                for u in c.as_chunks::<4>().0 {
                    let cp = u32::from_le_bytes(*u);
                    if cp == 0 {
                        break; // numpy pads with NULs
                    }
                    s.push(char::from_u32(cp).ok_or("invalid code point in npy string")?);
                }
                v.push(s);
            }
            NpyData::Str(v)
        }
    };
    let mut arr = NpyArray {
        shape: h.shape,
        data,
    };
    if h.fortran && arr.shape.len() == 2 {
        arr = transpose_fortran(arr);
    } else if h.fortran && arr.shape.len() > 2 {
        return Err("Fortran-order arrays with >2 dimensions are not supported".into());
    }
    Ok(arr)
}

fn transpose_fortran(a: NpyArray) -> NpyArray {
    let (r, c) = (a.shape[0], a.shape[1]);
    fn t<T: Clone>(v: &[T], r: usize, c: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(v.len());
        for i in 0..r {
            for j in 0..c {
                out.push(v[j * r + i].clone());
            }
        }
        out
    }
    let data = match &a.data {
        NpyData::U8(v) => NpyData::U8(t(v, r, c)),
        NpyData::I64(v) => NpyData::I64(t(v, r, c)),
        NpyData::F64(v) => NpyData::F64(t(v, r, c)),
        NpyData::Str(v) => NpyData::Str(t(v, r, c)),
    };
    NpyArray {
        shape: a.shape,
        data,
    }
}

/// Load a standalone `.npy` file.
pub fn load_npy(path: &Path) -> Result<NpyArray, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let size = f
        .metadata()
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    read_npy(&mut f, size).map_err(|e| format!("{}: {e}", path.display()))
}

/// An opened `.npz` archive (member index only; members load on demand).
pub struct Npz {
    path: PathBuf,
    file: File,
    /// member name (without `.npy`) -> (data offset, size)
    members: HashMap<String, (u64, u64)>,
}

impl Npz {
    pub fn open(path: &Path) -> Result<Npz, String> {
        let ctx = |e: String| format!("{}: {e}", path.display());
        let mut file = File::open(path).map_err(|e| ctx(e.to_string()))?;
        let members = Self::index(&mut file).map_err(ctx)?;
        Ok(Npz {
            path: path.to_path_buf(),
            file,
            members,
        })
    }

    pub fn has(&self, name: &str) -> bool {
        self.members.contains_key(name)
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.members.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn read(&mut self, name: &str) -> Result<NpyArray, String> {
        let &(off, size) = self
            .members
            .get(name)
            .ok_or_else(|| format!("{}: no member {name:?}", self.path.display()))?;
        self.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        read_npy(&mut self.file, size).map_err(|e| format!("{} [{name}]: {e}", self.path.display()))
    }

    fn index(f: &mut File) -> Result<HashMap<String, (u64, u64)>, String> {
        let io = |e: std::io::Error| format!("read error: {e}");
        let flen = f.metadata().map_err(io)?.len();
        let tail_len = flen.min(22 + 65535);
        if tail_len < 22 {
            return Err("not a zip archive (too short)".into());
        }
        let mut tail = vec![0u8; tail_len as usize];
        f.seek(SeekFrom::Start(flen - tail_len)).map_err(io)?;
        f.read_exact(&mut tail).map_err(io)?;
        let eocd = (0..=tail.len() - 22)
            .rev()
            .find(|&i| le_u32(&tail, i) == 0x0605_4b50)
            .ok_or("not a zip archive (no end-of-central-directory record)")?;
        let mut n_total = le_u16(&tail, eocd + 10) as u64;
        let mut cd_size = le_u32(&tail, eocd + 12) as u64;
        let mut cd_off = le_u32(&tail, eocd + 16) as u64;
        if n_total == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_off == 0xFFFF_FFFF {
            // zip64: locator sits just before the EOCD
            if eocd < 20 || le_u32(&tail, eocd - 20) != 0x0706_4b50 {
                return Err("zip64 archive without a zip64 locator".into());
            }
            let z_off = le_u64(&tail, eocd - 20 + 8);
            if z_off.checked_add(56).is_none_or(|e| e > flen) {
                return Err("zip64 end record out of bounds".into());
            }
            let mut z = [0u8; 56];
            f.seek(SeekFrom::Start(z_off)).map_err(io)?;
            f.read_exact(&mut z).map_err(io)?;
            if le_u32(&z, 0) != 0x0606_4b50 {
                return Err("bad zip64 end record".into());
            }
            n_total = le_u64(&z, 32);
            cd_size = le_u64(&z, 40);
            cd_off = le_u64(&z, 48);
        }
        if n_total as usize > MAX_MEMBERS {
            return Err(format!("zip has too many members ({n_total})"));
        }
        if cd_size > MAX_CENTRAL_DIR || cd_off.checked_add(cd_size).is_none_or(|e| e > flen) {
            return Err("zip central directory out of bounds".into());
        }
        let mut cd = vec![0u8; cd_size as usize];
        f.seek(SeekFrom::Start(cd_off)).map_err(io)?;
        f.read_exact(&mut cd).map_err(io)?;
        let mut out = HashMap::new();
        let mut p = 0usize;
        for _ in 0..n_total {
            if p + 46 > cd.len() || le_u32(&cd, p) != 0x0201_4b50 {
                return Err("corrupt zip central directory".into());
            }
            let flags = le_u16(&cd, p + 8);
            let method = le_u16(&cd, p + 10);
            let mut csize = le_u32(&cd, p + 20) as u64;
            let mut usize_ = le_u32(&cd, p + 24) as u64;
            let nlen = le_u16(&cd, p + 28) as usize;
            let elen = le_u16(&cd, p + 30) as usize;
            let clen = le_u16(&cd, p + 32) as usize;
            let mut loff = le_u32(&cd, p + 42) as u64;
            let end = p + 46 + nlen + elen + clen;
            if end > cd.len() {
                return Err("corrupt zip central directory entry".into());
            }
            let name = std::str::from_utf8(&cd[p + 46..p + 46 + nlen])
                .map_err(|_| "non-UTF-8 zip member name".to_string())?
                .to_string();
            // zip64 extra field
            let extra = &cd[p + 46 + nlen..p + 46 + nlen + elen];
            let mut q = 0usize;
            while q + 4 <= extra.len() {
                let id = le_u16(extra, q);
                let sz = le_u16(extra, q + 2) as usize;
                if q + 4 + sz > extra.len() {
                    break;
                }
                if id == 0x0001 {
                    let mut r = q + 4;
                    let lim = q + 4 + sz;
                    for field in [&mut usize_, &mut csize, &mut loff] {
                        if *field == 0xFFFF_FFFF {
                            if r + 8 > lim {
                                return Err("truncated zip64 extra field".into());
                            }
                            *field = le_u64(extra, r);
                            r += 8;
                        }
                    }
                }
                q += 4 + sz;
            }
            p = end;
            if flags & 1 != 0 {
                return Err(format!("zip member {name:?} is encrypted"));
            }
            if method != 0 {
                return Err(format!(
                    "zip member {name:?} is compressed (method {method}); only uncompressed \
                     np.savez archives are supported"
                ));
            }
            if csize != usize_ {
                return Err(format!("zip member {name:?}: stored size mismatch"));
            }
            // local header -> data offset
            if loff.checked_add(30).is_none_or(|e| e > flen) {
                return Err(format!("zip member {name:?}: local header out of bounds"));
            }
            let mut lh = [0u8; 30];
            f.seek(SeekFrom::Start(loff)).map_err(io)?;
            f.read_exact(&mut lh).map_err(io)?;
            if le_u32(&lh, 0) != 0x0403_4b50 {
                return Err(format!("zip member {name:?}: bad local header"));
            }
            let data_off = loff + 30 + le_u16(&lh, 26) as u64 + le_u16(&lh, 28) as u64;
            if data_off.checked_add(csize).is_none_or(|e| e > flen) {
                return Err(format!(
                    "zip member {name:?}: data out of bounds (truncated file?)"
                ));
            }
            let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
            out.insert(key, (data_off, csize));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_blob(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
        let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(' ');
        }
        h.push('\n');
        let mut v = b"\x93NUMPY\x01\x00".to_vec();
        v.extend_from_slice(&(h.len() as u16).to_le_bytes());
        v.extend_from_slice(h.as_bytes());
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn parses_and_rejects() {
        let data: Vec<u8> = [1i64, -2, 3].iter().flat_map(|x| x.to_le_bytes()).collect();
        let b = npy_blob("<i8", "(3,)", &data);
        let a = read_npy(&mut &b[..], b.len() as u64).unwrap();
        assert_eq!(a.data, NpyData::I64(vec![1, -2, 3]));
        // truncated payload
        assert!(read_npy(&mut &b[..b.len() - 1], b.len() as u64 - 1).is_err());
        // shape claims more than present
        let b2 = npy_blob("<i8", "(4,)", &data);
        assert!(read_npy(&mut &b2[..], b2.len() as u64).is_err());
        // absurd shape must not allocate
        let b3 = npy_blob("<i8", "(100000000000000,)", &data);
        assert!(read_npy(&mut &b3[..], b3.len() as u64).is_err());
        // overflowing shape
        let b4 = npy_blob("<i8", "(18446744073709551615, 4)", &data);
        assert!(read_npy(&mut &b4[..], b4.len() as u64).is_err());
        // unsupported dtype
        let b5 = npy_blob(">i8", "(3,)", &data);
        assert!(read_npy(&mut &b5[..], b5.len() as u64).is_err());
        // unicode scalar
        let s: Vec<u8> = "ab"
            .chars()
            .flat_map(|c| (c as u32).to_le_bytes())
            .chain([0u8; 4])
            .collect();
        let b6 = npy_blob("<U3", "()", &s);
        let a6 = read_npy(&mut &b6[..], b6.len() as u64).unwrap();
        assert_eq!(a6.scalar_str().unwrap(), "ab");
        // garbage
        assert!(read_npy(&mut &b"hello world!"[..], 12).is_err());
    }
}
