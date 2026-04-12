---
title: LegalReviewEnv
emoji: ⚖
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
---

# LegalReviewEnv — OpenEnv Hackathon

OpenEnv-compliant AI legal contract review environment.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/reset` | Reset environment |
| POST | `/step` | Take action |
| GET | `/state` | Get state |
| POST | `/validate` | Run full episode |

## Tasks
- **easy**: 10-clause NDA
- **medium**: 50-clause SaaS Agreement  
- **hard**: 120-clause M&A Due Diligence
