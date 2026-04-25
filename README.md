# Workflow-CI — Titanic MLflow Project

**Nama:** Fauzan Aidil Luthfi  
**NIM:** apc352d6y0439

## Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                     ← GitHub Actions CI (Advanced)
├── MLProject/
│   ├── modelling.py                   ← Script training
│   ├── conda.yaml                     ← Environment dependencies
│   ├── MLProject                      ← MLflow Project config
│   ├── Dockerfile                     ← Untuk build Docker image
│   └── titanic_preprocessing/
│       └── train_preprocessed.csv    ← Dataset hasil preprocessing
├── saved_artifacts/                   ← Artefak hasil CI (auto-generated)
└── README.md
```

## Cara Setup Secrets di GitHub

Masuk ke repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Tambahkan secrets berikut:

| Secret Name | Isi |
|---|---|
| `DOCKERHUB_USERNAME` | username Docker Hub kamu |
| `DOCKERHUB_TOKEN` | Access Token dari Docker Hub |
| `MLFLOW_TRACKING_URI` | URL DagsHub MLflow (dari Kriteria 2) |
| `MLFLOW_TRACKING_USERNAME` | username DagsHub |
| `MLFLOW_TRACKING_PASSWORD` | password / token DagsHub |

## Cara Jalankan Manual

1. Buka tab **Actions** di GitHub
2. Klik **Workflow CI — Titanic MLflow + Docker**
3. Klik **Run workflow** → **Run workflow**


## Docker Hub

Image tersedia di:
```
docker pull username_kamu/titanic-mlflow:latest
```

## Cara Dapat Docker Hub Token

1. Login ke https://hub.docker.com
2. Klik avatar → **Account Settings** → **Security**
3. Klik **New Access Token**
4. Kasih nama → **Generate**
5. Copy token → paste ke GitHub Secrets sebagai `DOCKERHUB_TOKEN`
