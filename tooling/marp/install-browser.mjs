// Explicit cloud setup; ordinary build/check never downloads a browser.
import { install, Browser } from '@puppeteer/browsers';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const browser = await install({
  browser: Browser.CHROME,
  buildId: puppeteer.browserVersion,
  cacheDir: fileURLToPath(new URL('./node_modules/.cache/chrome', import.meta.url)),
  installDeps: process.platform === 'linux' && process.getuid?.() === 0,
});
console.log(`Installed presentation browser: ${browser.executablePath}`);
