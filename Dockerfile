FROM node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install --legacy-peer-deps

COPY . .
RUN npm run build

ENV NODE_ENV=production
ENV PORT=3000
EXPOSE 3000

CMD ["sh", "-c", "if [ \"$WORKER_MODE\" = \"1\" ] || [ \"$WORKER_MODE\" = \"true\" ]; then exec node --max-old-space-size=128 dist-server/workerEntry.mjs; else exec node --max-old-space-size=384 dist-server/server.mjs; fi"]
