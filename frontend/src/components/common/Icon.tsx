import type { CSSProperties } from "react";

/**
 * Google Material Symbols Outlined -- the exact icon system used by every approved
 * Stitch reference (loaded in index.html). `name` is a literal Material Symbols
 * ligature name (e.g. "dashboard", "terminal", "warning") -- see
 * fonts.google.com/icons for the full set.
 */
export function Icon({
  name,
  size = 18,
  filled = false,
  className,
  style,
}: {
  name: string;
  size?: number;
  filled?: boolean;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <span
      className={`material-symbols-outlined${className ? ` ${className}` : ""}`}
      style={{
        fontSize: size,
        width: size,
        height: size,
        fontVariationSettings: `'FILL' ${filled ? 1 : 0}, 'wght' 400, 'GRAD' 0, 'opsz' 24`,
        ...style,
      }}
      aria-hidden="true"
    >
      {name}
    </span>
  );
}
