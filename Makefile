.PHONY: all download_data parse_data train inference evaluate clean setup

download_data:
	@echo "Downloading CADETS dataset..."
	@mkdir -p data/cadets/raw_data
	@cd data/cadets/raw_data && \
		curl "https://drive.google.com/uc?id=1AcWrYiBmgAqp7DizclKJYYJJBQbnDMfb" -o cadets_json_1.tar.gz && \
		curl "https://drive.google.com/uc?id=1XLCEhf5DR8xw3S-Fimcj32IKnfzHFPJW" -o cadets_json_2.tar.gz && \
		curl "https://drive.google.com/uc?id=1EycO23tEvZVnN3VxOHZ7gdbSCwqEZTI1" -o cadets_json_3.tar.gz && \
		tar -xvf cadets_json_1.tar.gz && \
		tar -xvf cadets_json_2.tar.gz && \
		tar -xvf cadets_json_3.tar.gz && \
		rm -f *.tar.gz

parse_data: 
	@echo "Parsing CADETS data..."
	python -m src.parsers.cadets_parser cadets

train: 
	@echo "Training models..."
	python -m src.training.train_embeddings
	python -m src.training.train_detector

inference:
	@echo "Running inference..."
	python -m src.inference.detect_anomalies
	python -m src.inference.trace_alerts

evaluate:
	@echo "Evaluating results..."
	python -m src.evaluation.evaluate
