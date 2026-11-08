// Render sequence diagrams before delivery; the presentation needs no Mermaid runtime.
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import puppeteer from 'puppeteer-core';

const require = createRequire(import.meta.url);
const fontRoot = dirname(require.resolve('@fontsource/noto-sans-tc/package.json'));
const fontLicense = (await readFile(resolve(fontRoot, 'LICENSE'), 'utf8'))
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

async function embeddedFont(source) {
  const points = [...new Set([...source].map((character) => character.codePointAt(0)))];
  const css = await readFile(resolve(fontRoot, '400.css'), 'utf8');
  const faces = [];
  for (const [face] of css.matchAll(/@font-face\s*\{[^}]+\}/g)) {
    const ranges = [...face.matchAll(/U\+([\da-f]+)(?:-([\da-f]+))?/gi)];
    if (!ranges.some(([, first, last]) => points.some((point) =>
      point >= parseInt(first, 16) && point <= parseInt(last ?? first, 16)))) continue;
    const [, path] = face.match(/url\(([^)]+\.woff2)\)/);
    const font = await readFile(resolve(fontRoot, path));
    faces.push(face.replace(/src:[^;]+;/,
      `src: url(data:font/woff2;base64,${font.toString('base64')}) format('woff2');`));
  }
  // Compact CJK line boxes without scaling or distorting the glyphs.
  return faces.join('\n')
    .replaceAll('font-display: swap;', 'font-display: block; '
      + 'ascent-override: 90%; descent-override: 10%; line-gap-override: 0%;')
    .replaceAll('Noto Sans TC', 'TradingDev Diagram');
}

const sources = JSON.parse(await readFile(process.argv[2], 'utf8'));
for (const [index, source] of sources.entries()) {
  if (!/^\s*sequenceDiagram\b/.test(source) || /%%\s*\{/.test(source)) {
    throw new Error(`Mermaid diagram ${index + 1}: use sequenceDiagram without configuration directives.`);
  }
}

let browser;
try {
  try {
    browser = await puppeteer.launch({
      ...(process.env.TRADINGDEV_PRESENTATION_BROWSER
        ? { executablePath: process.env.TRADINGDEV_PRESENTATION_BROWSER }
        : { channel: 'chrome' }),
      headless: true,
    });
  } catch (error) {
    throw new Error('Mermaid rendering requires Chrome. Install Chrome or set '
      + 'TRADINGDEV_PRESENTATION_BROWSER to a Chromium executable.', { cause: error });
  }
  const page = await browser.newPage();
  await page.setRequestInterception(true);
  page.on('request', (request) => request.url().startsWith('data:')
    ? request.continue() : request.abort());
  const fontCSS = await embeddedFont(sources.join('\n'));
  await page.setContent(`<style>${fontCSS}</style>`);
  await page.addScriptTag({ path: require.resolve('mermaid/dist/mermaid.min.js') });
  await page.evaluate(async (text) => {
    await document.fonts.load('400 24px "TradingDev Diagram"', text);
    await document.fonts.ready;
  }, sources.join('\n'));

  const diagrams = [];
  for (const [index, source] of sources.entries()) {
    const id = `tradingdev-mermaid-${index + 1}`;
    try {
      const svg = await page.evaluate(async ({ source, id }) => {
        mermaid.initialize({
          startOnLoad: false,
          fontSize: 32,
          securityLevel: 'strict',
          deterministicIds: true,
          deterministicIDSeed: id,
          fontFamily: 'TradingDev Diagram',
          theme: 'base',
          themeVariables: {
            fontFamily: 'TradingDev Diagram',
            fontSize: '32px',
            primaryColor: '#f8f7f3',
            primaryTextColor: '#16343c',
            actorBkg: '#f8f7f3',
            actorBorder: '#c9d4d2',
            actorTextColor: '#16343c',
            actorLineColor: '#a7bfba',
            signalColor: '#087f79',
            signalTextColor: '#16343c',
          },
          sequence: {
            useMaxWidth: false,
            mirrorActors: false,
            actorMargin: 360,
            width: 260,
            height: 48,
            diagramMarginX: 16,
            diagramMarginY: 4,
            boxMargin: 0,
          },
        });
        const { svg } = await mermaid.render(id, source);
        const document = new DOMParser().parseFromString(svg, 'image/svg+xml');
        const ids = new Map([...document.querySelectorAll('[id]')].map((element) =>
          [element.id, element.id === id ? id : `${id}-${element.id}`]));
        const references = (value) => value.replace(/url\(#([^)]+)\)/g,
          (match, target) => ids.has(target) ? `url(#${ids.get(target)})` : match);
        for (const element of document.querySelectorAll('*')) {
          for (const attribute of element.attributes) {
            if (attribute.name === 'id') attribute.value = ids.get(attribute.value);
            else if (['aria-labelledby', 'aria-describedby'].includes(attribute.name)) {
              attribute.value = attribute.value.split(' ').map((target) =>
                ids.get(target) ?? target).join(' ');
            } else if (['href', 'xlink:href'].includes(attribute.name)
              && attribute.value.startsWith('#')) {
              attribute.value = `#${ids.get(attribute.value.slice(1)) ?? attribute.value.slice(1)}`;
            } else attribute.value = references(attribute.value);
          }
          if (element.localName === 'style') {
            element.textContent = references(element.textContent).replace(
              /#([\w-]+)/g, (match, target) => ids.has(target) ? `#${ids.get(target)}` : match);
          }
        }
        return new XMLSerializer().serializeToString(document.documentElement);
      }, { source, id });
      // Font faces are shared by every inline diagram in this HTML document.
      diagrams.push(index === 0
        ? svg.replace(/(<svg\b[^>]*>)/,
          `$1<metadata>${fontLicense}</metadata><style>${fontCSS}</style>`)
        : svg);
    } catch (error) {
      throw new Error(`Mermaid diagram ${index + 1}: ${error.message}`, { cause: error });
    }
  }
  await writeFile(process.argv[3], JSON.stringify(diagrams));
} finally {
  if (browser) await browser.close();
}
