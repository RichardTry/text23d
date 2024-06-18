# Text to 3D

## Install project

```
git clone https://github.com/RichardTry/text23d

cd text23d
```

### Via Docker Compose

#### Set Telegram bot token at .env file
```
BOT_TOKEN="<BOT_TOKEN>"
```
#### Run Docker Compose
```
docker compose up -d
```

### Locally

#### Create environment

```
python -m venv venv
source venv/bin/activate
```

#### Install requirements

```
pip install -r requirements.txt
pip install -r heavy_requirements.txt
```

#### Run the project

```
python3 main.py
```
