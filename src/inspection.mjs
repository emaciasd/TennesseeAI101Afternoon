export function classify(probabilities, correctClass, incorrectClass, threshold) {
  if (!probabilities.length || !correctClass || !incorrectClass || correctClass === incorrectClass) return { verdict: 'REVIEW', confidence: 0, label: 'Unconfigured' };
  const best = probabilities.reduce((a, b) => b.probability > a.probability ? b : a);
  return { verdict: best.probability < threshold ? 'REVIEW' : best.className === correctClass ? 'GO' : best.className === incorrectClass ? 'NO-GO' : 'REVIEW', confidence: best.probability, label: best.className };
}
export function modelBase(value) {
  const url = new URL(value.trim());
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(url.hostname))) throw new Error('Use an HTTPS model URL.');
  url.search = ''; url.hash = '';
  url.pathname = url.pathname.replace(/(?:model|metadata)\.json$/, '').replace(/\/?$/, '/');
  return url.href;
}
export function csv(records) {
  const cell = value => '"' + String(value).replace(/^[=+@\-\t\r]/, "'$&").replaceAll('"', '""') + '"';
  return [['timestamp','verdict','confidence','class','model_url'], ...records.map(r => [new Date(r.time).toISOString(),r.verdict,(r.confidence * 100).toFixed(2),r.label,r.modelUrl])].map(row => row.map(cell).join(',')).join('\r\n');
}
