echo "exporting fred api key"
source scripts/env.sh

echo "downloading return data"
python scripts/download.py

echo "processing risk models"
python scripts/risk.py
