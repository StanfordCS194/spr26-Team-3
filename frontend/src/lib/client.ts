/**
 * Typed API client. Once `just typegen` has run, replace the `any` schema
 * with the generated `paths` type from `@worldscan/shared/ts/api`.
 */
import createClient from "openapi-fetch";

// TODO(PR-A.10): after first run of `just typegen`, swap this to:
//   import type { paths } from "@worldscan/shared/ts/api";
//   export const api = createClient<paths>({ baseUrl: "" });
// For PR-A scaffold we use an untyped client so the app builds before
// the backend has been started once.
type Paths = Record<string, unknown>;

export const api = createClient<Paths>({ baseUrl: "" });
