.PHONY: build up down exec test clean

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

exec:
	docker compose exec hierdamap bash

train:
	docker compose exec hierdamap python3 train_our_mapping_b2s.py

test:
	docker compose exec hierdamap python3 test_hierdamap.py

clean:
	docker compose down -v
