import random
import os
import shutil

train_dir = "../data/train"
test_dir = "../data/test"

def prep_test_data(pokemon, train_dir, test_dir):
    # Rutas de este Pokémon
    pokemon_train_dir = os.path.join(train_dir, pokemon)
    pokemon_test_dir = os.path.join(test_dir, pokemon)

    # Crear carpeta en test si no existe
    os.makedirs(pokemon_test_dir, exist_ok=True)

    # Listar SOLO archivos (no directorios)
    all_items = os.listdir(pokemon_train_dir)
    files = [
        f for f in all_items
        if os.path.isfile(os.path.join(pokemon_train_dir, f))
    ]

    if len(files) == 0:
        print(f"[WARN] {pokemon}: no se encontraron imágenes en train.")
        return

    # Tomar hasta 15 imágenes (o menos si no hay tantas)
    n_sample = 15 if len(files) >= 15 else len(files)
    test_data = random.sample(files, n_sample)

    print(f"{pokemon}: moviendo {n_sample} imágenes a test.")
    # print(test_data)  # si quieres ver cuáles son

    for f in test_data:
        src = os.path.join(pokemon_train_dir, f)
        dst = os.path.join(pokemon_test_dir, f)
        try:
            shutil.copy(src, dst)
        except FileNotFoundError:
            # Si por ruta larga / OneDrive / etc. no se puede copiar, lo saltamos
            print(f"  [SKIP] No se pudo copiar: {src}")

if __name__ == "__main__":
    os.makedirs(test_dir, exist_ok=True)

    pokemons = [
        p for p in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, p))
    ]

    for pokemon in pokemons:
        prep_test_data(pokemon, train_dir, test_dir)

    print("test folder complete!!")
