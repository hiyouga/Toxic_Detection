# Toxic_Detection
 
![GitHub](https://img.shields.io/github/license/hiyouga/toxic_detection)

The technical report can be found [here](technical_report.pdf).

## Requirement

- Python >= 3.7
- torch >= 1.9.0
- numpy >= 1.17.2
- transformers >= 4.15.0

## Preparation

### Clone

```bash
git clone https://github.com/hiyouga/Toxic_Detection.git
```

### Create an anaconda environment:

```bash
conda create -n toxic python=3.7
conda activate toxic
pip install -r requirements.txt
```

## Usage

### Split data

```sh
python data/split_data.py
```

### Training

```sh
python main.py
```

### Show help message

```sh
python main.py -h
```

## Acknowledgements

This is a group homework for "Machine Learning" in BUAA Graduate School.

## Contact

hiyouga [AT] buaa [DOT] edu [DOT] cn

## License

MIT
