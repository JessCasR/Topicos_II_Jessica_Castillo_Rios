# Tópicos Selectos de Grandes Bases de Datos II

Proyecto de Machine Learning basado en el **WiDS Datathon 2024 Challenge 1** de Kaggle.

**Materia:** Tópicos Selectos de Grandes Bases de Datos II  
**Programa:** Maestría en Ciencia de Datos  
**Universidad:** Universidad de Guadalajara

## Descripción del Proyecto

Este repositorio contiene scripts modulares de Python para el análisis y modelado de datos del [WiDS Datathon 2024](https://www.kaggle.com/competitions/widsdatathon2024-challenge1).

## Estructura del Proyecto

```
Topicos_II/
├── data/           # Dataset del concurso
├── src/            # Scripts modulares de Python
├── notebooks/      # Jupyter notebooks para análisis exploratorio
└── README.md       # Este archivo
```

## Instalación y Configuración

### Descargar los Datos

1. Instala la API de Kaggle:
```bash
pip install kaggle
```

2. Configura tus credenciales de Kaggle siguiendo las [instrucciones oficiales](https://github.com/Kaggle/kaggle-api#api-credentials)

3. Descarga los datos del concurso:
```bash
kaggle competitions download -c widsdatathon2024-challenge1
```

4. Extrae los archivos en la carpeta `data/`:
```bash
mkdir -p data
unzip widsdatathon2024-challenge1.zip -d data/
```
