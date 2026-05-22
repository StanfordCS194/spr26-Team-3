/**
 * Imported at the very top of main.tsx so Sentry + PostHog init before
 * React renders. Both are no-ops if their env vars aren't set.
 */
import * as Sentry from "@sentry/react";
import posthog from "posthog-js";

const SENTRY_DSN = import.meta.env.VITE_SENTRY_DSN;
const POSTHOG_KEY = import.meta.env.VITE_POSTHOG_KEY;
const POSTHOG_HOST: string = import.meta.env.VITE_POSTHOG_HOST ?? "https://us.i.posthog.com";

if (SENTRY_DSN) {
  Sentry.init({ dsn: SENTRY_DSN, tracesSampleRate: 0.1 });
}

if (POSTHOG_KEY) {
  posthog.init(POSTHOG_KEY, { api_host: POSTHOG_HOST });
}
