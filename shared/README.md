# @worldscan/shared

Generated OpenAPI types consumed by the frontend.

- `openapi.json` — written by `just typegen` from the running backend's
  `/openapi.json`.
- `ts/api.d.ts` — written by `yarn workspace @worldscan/shared run codegen`
  (which the `just typegen` task chains).

The frontend imports types from here as:

```ts
import type { paths } from "@worldscan/shared/ts/api";
```

These files are git-ignored (`shared/ts/api.d.ts`) because they're derived
artifacts. The Husky pre-commit hook keeps them up to date.
