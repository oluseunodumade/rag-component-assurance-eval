export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function score(value: number, digits = 2): string {
  return value.toFixed(digits);
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function formatCategory(category: string): string {
  return category.replaceAll("_", " ");
}

export function formatRoute(route: string): string {
  return route.replaceAll("_", " ");
}
