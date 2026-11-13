// One authoring setting supplies the cover QR image and both destination links.
import { decodeHTML, escapeAttribute } from 'entities';
import { Marp } from '@marp-team/marp-core';
import QRCode from 'qrcode';

const qrMarker = '{{presentation-qr}}';
const linkMarker = '{{presentation-link}}';

export function presentationSource(markdown) {
  const declarations = [...markdown.matchAll(/^<!--[ \t]*presentation-url\b[^\r\n]*/gm)];
  if (!declarations.length && !markdown.includes(qrMarker) && !markdown.includes(linkMarker)) {
    return { markdown, url: null };
  }
  const setting = markdown.trimStart().match(/^<!--[ \t]*presentation-url:[ \t]*(.*?)[ \t]*-->[ \t]*(?:\r?\n|$)/);
  if (declarations.length !== 1 || !setting) {
    throw new Error('Declare exactly one <!-- presentation-url: https://… --> on the first nonblank line.');
  }
  const value = setting[1];
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error('presentation-url must be an absolute HTTP(S) URL.');
  }
  if (!/^https?:\/\//i.test(value) || /[\s\\\u0000-\u001f\u007f]/u.test(value)
      || !['https:', 'http:'].includes(parsed.protocol) || !parsed.hostname
      || parsed.username || parsed.password) {
    throw new Error('presentation-url must be an absolute HTTP(S) URL without credentials, whitespace or backslashes.');
  }
  // Resolve the footer before Marp copies it into section metadata and CSS.
  const footer = [...markdown.matchAll(/<!--[\s\S]*?-->/g)]
    .find(([text]) => /^<!--\s*_?footer:/.test(text) && text.includes(linkMarker));
  if (markdown.split(linkMarker).length !== 2 || !footer) {
    throw new Error('Use {{presentation-link}} once in a footer directive.');
  }
  const label = parsed.href.replace(/^https?:\/\//, '')
    .replaceAll('&', '&amp;')
    .replace(/[\\`*{}\[\]()#+\-.!_|~<>]/g, '\\$&');
  // Markdown decodes entities in link destinations as well as in visible text.
  const link = `[${label}](<${parsed.href.replaceAll('&', '&amp;')}>)`;
  // Remove our setting before Marp can interpret it as a speaker note.
  return {
    markdown: markdown.trimStart().slice(setting[0].length)
      .replace(footer[0], () => footer[0].replace(linkMarker, () => link)),
    url: parsed.href,
  };
}

export async function renderPresentationUrl(content, url) {
  if (url === null) return content;
  // Recognize Marp's encoded destination, then use the same WHATWG URL as the QR.
  // In particular, Markdown encodes [] in queries and even in IPv6 hosts.
  const markdownDestination = new Marp().markdown.normalizeLink(url);
  const destination = escapeAttribute(url);
  let linked = false;
  content = content.replace(/<footer\b[^>]*>[\s\S]*?<\/footer>/g, (footer) =>
    footer.replace(/(<a\b[^>]*\shref=")([^"]*)"/g, (anchor, prefix, href) => {
      if (decodeHTML(href) !== markdownDestination) return anchor;
      linked = true;
      return `${prefix}${destination}"`;
    }));
  if (!linked) {
    throw new Error('The presentation URL must produce a matching rendered footer link; use a footer block scalar (|-).');
  }
  const qrParagraph = `<p>${qrMarker}</p>`;
  if (content.split(qrMarker).length !== 2 || !content.includes(qrParagraph)) {
    throw new Error('Use {{presentation-qr}} once as its own paragraph.');
  }
  const image = await QRCode.toDataURL(url, {
    type: 'image/png', errorCorrectionLevel: 'Q', margin: 4, scale: 12,
  });
  return content
    .replace(qrParagraph, () => `<p><a href="${destination}"><img class="presentation-qr" src="${image}" alt="簡報 QR code"></a></p>`);
}
