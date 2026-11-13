// Shared compiler for local previews and source-only cloud deployments.
import { readFile, rename, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { marpCli } from '@marp-team/marp-cli';
import { decodeHTML } from 'entities';
import { renderDiagrams } from './render-diagrams.mjs';
import { presentationSource, renderPresentationUrl } from './presentation-url.mjs';

const require = createRequire(import.meta.url);
const [source, theme, output] = process.argv.slice(2).map((path) => resolve(path));
if (!source || !theme || !output || process.argv.length !== 5) {
  throw new Error('Usage: node build.mjs slides.md theme.css output.html');
}
const temporary = `${output}.building-${process.pid}`;
const prepared = `${output}.source-${process.pid}.md`;
try {
  const { markdown, url } = presentationSource(await readFile(source, 'utf8'));
  await writeFile(prepared, markdown);
  const status = await marpCli([
    '--no-config', '--no-html', '--engine',
    require.resolve('@marp-team/marp-core/lib/marp.js'),
    prepared, '--theme-set', theme, '--theme', 'tradingdev',
    '--title', 'TradingDev｜能學習策略開發經驗、自我演進的研究後端',
    '--description', '產品介紹、使用場景、分類比較與後續開發方向',
    '--output', temporary,
  ]);
  if (status !== 0) throw new Error(`Marp failed (${status}).`);
  let content = await renderPresentationUrl(await readFile(temporary, 'utf8'), url);
  const fence = /<pre\b[^>]*><code class="language-mermaid">(.*?)<\/code><\/pre>/gs;
  const sources = [...content.matchAll(fence)].map((match) => decodeHTML(match[1]));
  if (sources.length) {
    const diagrams = await renderDiagrams(sources);
    let index = 0;
    content = content.replace(fence, () =>
      `<div class="mermaid-diagram">${diagrams[index++]}</div>`);
  }
  await writeFile(temporary, content);
  await rename(temporary, output);
} finally {
  await Promise.all([temporary, prepared].map((path) => rm(path, { force: true })));
}
