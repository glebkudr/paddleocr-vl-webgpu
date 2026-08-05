# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected security vulnerability.
Report it to `support@sotaocr.com` with reproduction details and the affected
version or commit.

## Image privacy

The included example performs inference in the browser. It does not send image bytes
to SotaOCR. The browser still downloads public model files and normal static
site assets from their configured origins. Deployers should review those
origins, set an appropriate Content Security Policy, and serve the application
over HTTPS.

Never place credentials in frontend environment variables or committed files.
Anything shipped to a browser must be treated as public.
