/**
 * Largest-Triangle-Three-Buckets downsampling (Steinarsson 2013).
 *
 * A century of daily closes is ~24,700 points. At 1920 CSS pixels that is
 * thirteen points per pixel: the detail is not merely expensive to render, it is
 * not visible. So the series is reduced before it leaves the Worker, where the
 * cost is paid once rather than by every reader's main thread.
 *
 * **Why not every Nth point.** Stride sampling is one line and it is wrong here
 * in a way that matters: a crash is a single-day event. 1987-10-19 is one
 * observation, and every-Nth-point drops it for any N > 1. The result is a chart
 * of a century of markets with Black Monday missing — which is worse than
 * honestly showing five years, because it looks complete.
 *
 * LTTB keeps it. The algorithm walks fixed buckets and, from each, keeps the
 * point forming the largest triangle with the previously kept point and the next
 * bucket's centroid. A one-day collapse is exactly the point that maximises that
 * area, so spikes are what the method preserves first and flat stretches are
 * what it spends its budget on last.
 *
 * First and last points are always kept, so the visible span is never shortened
 * by decimation — a chart that quietly starts later than the data does is the
 * same bug as a truncated response.
 */

export interface Point {
  /** Milliseconds since epoch. Numeric so triangle areas are meaningful. */
  x: number;
  y: number;
}

/** Below this, decimation is skipped and the raw window is returned. */
export const MIN_THRESHOLD = 3;

/**
 * Reduce `data` to at most `threshold` points, preserving visual shape.
 *
 * `data` must be sorted ascending by `x`; the caller reads it out of SQLite in
 * that order. Non-finite `y` values should be filtered out first — they cannot
 * contribute a triangle area and would poison a bucket's centroid.
 */
export function lttb<T extends Point>(data: T[], threshold: number): T[] {
  if (threshold >= data.length || threshold < MIN_THRESHOLD) return data;

  const sampled: T[] = [data[0]!];
  // Buckets span the interior only: the endpoints are kept unconditionally.
  const every = (data.length - 2) / (threshold - 2);
  let anchor = 0;

  for (let i = 0; i < threshold - 2; i++) {
    // Centroid of the *next* bucket — the triangle's third vertex.
    const nextStart = Math.floor((i + 1) * every) + 1;
    const nextEnd = Math.min(Math.floor((i + 2) * every) + 1, data.length);
    let avgX = 0;
    let avgY = 0;
    const span = Math.max(nextEnd - nextStart, 1);
    for (let j = nextStart; j < nextEnd; j++) {
      avgX += data[j]!.x;
      avgY += data[j]!.y;
    }
    avgX /= span;
    avgY /= span;

    const rangeStart = Math.floor(i * every) + 1;
    const rangeEnd = Math.min(Math.floor((i + 1) * every) + 1, data.length);
    const anchorPoint = data[anchor]!;

    let maxArea = -1;
    let chosen = rangeStart;
    for (let j = rangeStart; j < rangeEnd; j++) {
      const point = data[j]!;
      const area =
        Math.abs(
          (anchorPoint.x - avgX) * (point.y - anchorPoint.y) -
            (anchorPoint.x - point.x) * (avgY - anchorPoint.y),
        ) * 0.5;
      if (area > maxArea) {
        maxArea = area;
        chosen = j;
      }
    }

    sampled.push(data[chosen]!);
    anchor = chosen;
  }

  sampled.push(data[data.length - 1]!);
  return sampled;
}

/**
 * LTTB with the series' global extremes forced in.
 *
 * LTTB is shape-preserving but makes no *guarantee* about any particular
 * observation, and the acceptance criterion here is specifically that certain
 * dates survive. Re-inserting the highest and lowest points costs at most two
 * samples and turns "very likely kept" into "kept", which is the difference
 * between a property and a hope when the point in question is a market crash.
 *
 * The result stays sorted, so callers can treat it as an ordinary series.
 */
export function decimate<T extends Point>(data: T[], threshold: number): T[] {
  if (threshold >= data.length || threshold < MIN_THRESHOLD) return data;

  const sampled = lttb(data, threshold);
  const kept = new Set(sampled.map((p) => p.x));

  let lowest = data[0]!;
  let highest = data[0]!;
  for (const point of data) {
    if (point.y < lowest.y) lowest = point;
    if (point.y > highest.y) highest = point;
  }

  const missing = [lowest, highest].filter((p) => !kept.has(p.x));
  if (missing.length === 0) return sampled;

  return [...sampled, ...missing].sort((a, b) => a.x - b.x);
}
