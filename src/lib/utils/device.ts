/** Returns true when the app should refuse to run (phones / tablets). */
export function isUnsupportedDevice(): boolean {
  const uaData = (navigator as Navigator & { userAgentData?: { mobile?: boolean } }).userAgentData;
  if (uaData?.mobile === true) return true;

  const ua = navigator.userAgent;
  if (/Android|iPhone|iPod|iPad|Mobile/i.test(ua)) return true;

  // iPadOS 13+ often reports as Macintosh with touch
  if (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1) return true;

  return false;
}
