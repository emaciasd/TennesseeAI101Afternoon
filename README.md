# PartLens

A browser-based GO/NO-GO gage for visual part inspection, using a Teachable Machine **image** model. TypeScript, React, and TensorFlow.js. Camera frames and inference stay in the visitor's browser; no backend or API key is needed. The browser downloads model files from the URL you provide. Fonts are fetched from Google Fonts, with system-font fallbacks.

## Develop

Requires Node.js 22.12+ (or 24).

```sh
npm ci
npm run dev
```

```sh
npm test
npm run build
npm run preview
```

`npm run test:browser` runs an integration test against the running development server, using Chromium at `/usr/bin/chromium` or the `CHROMIUM_PATH` override. It uses a synthetic camera and a generated model fixture, not production part training data.

## Connect your model

1. Train an image project in Teachable Machine with examples of correct and incorrect parts. Include representative lighting, angles, backgrounds, and defects; an optional third empty/background class helps avoid classifying an empty station as a part.
2. Choose **Export Model → TensorFlow.js → Upload my model** and copy its model URL. You can also host the exported `model.json`, `metadata.json`, and weights together on an HTTPS host that allows cross-origin browser requests.
3. Paste the URL into Station settings and select **Load**. Explicitly verify which labels map to GO and NO-GO. Other classes and low-confidence predictions become REVIEW.
4. Allow camera access. The model sees a square center crop of the camera frame. Position one part at a time inside the guide; match training conditions.
5. Set a confidence threshold, then record each inspection manually. Optional timed recording samples every 1–30 seconds and can count the same part more than once. Counters measure observations, not unique parts. REVIEW is excluded from the acceptance rate.
6. Export CSV for records. The latest 1,000 observations persist in that browser's local storage. History is not shared across users/devices; clearing browser data removes it. CSV includes the timestamp, result, confidence, winning class, and model URL. Images are never recorded.

Share a model-prefilled link by adding `?model=` followed by the URL-encoded model address to the app address. The class mappings must still be checked by each operator. The confidence score is the model's output, not a calibrated measure of manufacturing quality: validate your trained model on representative held-out correct and defective parts before relying on it for acceptance decisions.

## Public deployment

The static build works on GitHub Pages, Netlify, Vercel, or any HTTPS static host. HTTPS is required for camera permission, except on localhost. Visitors need their own camera and must grant browser permission. No software installation is needed for visitors.

The included GitHub Actions workflow deploys on pushes to `main`. In GitHub repository **Settings → Pages**, select **GitHub Actions** as the source, then push these files to `main` or run the deployment workflow. If deployment succeeds, the expected address is `https://emaciasd.github.io/TennesseeAI101Afternoon/`; this address is not live until deployment completes. A private repository may require a GitHub plan that supports Pages or another static host.

For other hosts, run `npm run build` and publish `dist/`. For GitHub Pages, use `npm run build -- --base=/TennesseeAI101Afternoon/` as in the workflow. No environment secrets are required by the application.
