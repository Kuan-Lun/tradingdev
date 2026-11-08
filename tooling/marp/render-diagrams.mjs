// Render sequence diagrams before delivery; the presentation needs no Mermaid runtime.
import { createReadStream } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { createBrotliDecompress } from 'node:zlib';
import puppeteer from 'puppeteer-core';
import { extract } from 'tar-fs';

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

export async function renderDiagrams(sources) {
  for (const [index, source] of sources.entries()) {
    if (!/^\s*sequenceDiagram\b/.test(source) || /%%\s*\{/.test(source)) {
      throw new Error(`Mermaid diagram ${index + 1}: use sequenceDiagram without configuration directives.`);
    }
  }

  let browser;
  let cloudTemporary;
  const previousTemporary = process.env.TMPDIR;
  try {
    const executable = process.env.TRADINGDEV_PRESENTATION_BROWSER;
    let options = {
      ...(executable ? { executablePath: executable } : { channel: 'chrome' }),
      headless: true,
    };
    if (process.env.WORKERS_CI === '1') {
      if (process.platform !== 'linux' || process.arch !== 'x64') {
        throw new Error('The Workers Builds browser requires Linux x64.');
      }
      cloudTemporary = await mkdtemp(join(tmpdir(), 'tradingdev-cloud-browser-'));
      process.env.TMPDIR = cloudTemporary;
      const { default: chromium } = await import('@sparticuz/chromium');
      const binaries = resolve(dirname(require.resolve('@sparticuz/chromium')), '../../bin');
      // Workers Builds is not Lambda: explicitly extract the bundled libraries
      // instead of depending on the package's AWS environment detection.
      await pipeline(
        createReadStream(join(binaries, 'al2023.tar.br')),
        createBrotliDecompress(),
        extract(join(cloudTemporary, 'al2023')),
      );
      const executablePath = await chromium.executablePath();
      const fonts = join(cloudTemporary, 'fonts');
      // The bundled Lambda config points at /tmp/fonts. Our isolated extraction
      // needs paths relative to this config, including a private writable cache.
      await writeFile(join(fonts, 'fonts.conf'), `<?xml version="1.0"?>
<fontconfig>
  <dir prefix="relative">fonts</dir>
  <cachedir prefix="xdg">fontconfig</cachedir>
</fontconfig>`);
      options = {
        executablePath,
        args: chromium.args,
        headless: 'shell',
        env: {
          ...process.env,
          FONTCONFIG_PATH: fonts,
          FONTCONFIG_FILE: join(fonts, 'fonts.conf'),
          XDG_CACHE_HOME: join(cloudTemporary, 'cache'),
          LD_LIBRARY_PATH: [join(cloudTemporary, 'al2023/lib'), process.env.LD_LIBRARY_PATH]
            .filter(Boolean).join(':'),
        },
      };
    }
    try {
      browser = await puppeteer.launch(options);
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
    return diagrams;
  } finally {
    if (cloudTemporary) {
      if (previousTemporary === undefined) delete process.env.TMPDIR;
      else process.env.TMPDIR = previousTemporary;
    }
    try {
      if (browser) await browser.close();
    } catch (error) {
      if (cloudTemporary) throw new Error(
        `Browser shutdown is unverified; retained ${cloudTemporary}`, { cause: error });
      throw error;
    }
    if (cloudTemporary) await rm(cloudTemporary, { recursive: true, force: true });
  }
}
