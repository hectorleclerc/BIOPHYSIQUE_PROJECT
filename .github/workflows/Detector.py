#%% REORIENTATION ET CROP DES IMAGES

from pathlib import Path
from datetime import datetime
from PIL import Image
import shutil

# ========= À CONFIGURER =========
# Dossier où se trouvent les .tif d'origine
input_dir = Path(r"C:\Users\hecto\Documents\ENS\M2\Biophysique\Corentin\delapartdeCorentin_stitch")

# Dossier où tu veux écrire toutes les images (rotées + non rotées)
output_dir = Path(r"C:\Users\hecto\Documents\ENS\M2\Biophysique\Corentin\rotates")
output_dir.mkdir(parents=True, exist_ok=True)
# =================================


def parse_timestamp_from_name(path: Path) -> datetime:
    """
    Extrait un datetime à partir du début du nom de fichier, supposé être:
    YYYYMMDD_HHMM...
    Ex: '20241004_1040', '20241004_1040_toto', etc.
    """
    stem = path.stem  # nom sans extension
    # On prend les 13 premiers caractères: 'YYYYMMDD_HHMM'
    key = stem[:13]
    return datetime.strptime(key, "%Y%m%d_%H%M")


# --- Définition des bornes temporelles d'après ta consigne ---
t1 = datetime(2024, 10, 4, 10, 40)  # 20241004_1040
t2 = datetime(2024, 10, 8, 8, 20)   # 20241008_0820
t4 = datetime(2024, 10, 10, 12, 40) # 20241010_1240


def angle_for_timestamp(ts: datetime) -> float:
    """
    Retourne l'angle de rotation en degrés.
    Convention Pillow: angle > 0 -> rotation anti-horaire.
    """
    # 1) Avant 20241004_1040 : +1.2° anti-horaire
    if ts <= t1:
        return 1.22-1

    # 2) Entre 20241008_0820 (exclu) et 20241010_0942 (inclus) : -1.1° sens horaire
    if t2 <= ts < t4:
        return 1

    # 3) Après 20241010_1240 : +0.3° anti-horaire
    if ts >= t4:
        return 0.8-1

    # 4) Sinon : pas de rotation (image recopiée telle quelle)
    return 0.0-1


# --- Boucle principale sur les .tif ---
for tif_path in sorted(input_dir.glob("*.tif")):
    ts = parse_timestamp_from_name(tif_path)
    if ts>=t4 and ts>=t4:
        angle = angle_for_timestamp(ts)

        out_path = output_dir / tif_path.name
    
        if angle == 0.0:
            # Pas de rotation : on copie simplement le fichier
            shutil.copy2(tif_path, out_path)
            print(f"[COPIE] {tif_path.name} (pas de rotation)")
        else:
            # Rotation légère avec Pillow
            with Image.open(tif_path) as im:
                # expand=True pour éviter de rogner l'image.
                # Si tu veux garder strictement la même taille, mets expand=False
                rotated = im.rotate(angle, resample=Image.BICUBIC, expand=True)
                # Sauvegarde en TIFF
                rotated.save(out_path)
            print(f"[ROTATION {angle:+.1f}°] {tif_path.name} -> {out_path.name}")


#%% RECROP 

from pathlib import Path
from datetime import datetime
from PIL import Image
import shutil

# ================== PARAMÈTRES GÉNÉRAUX ==================
# Dossier d'entrée : images déjà horizontales (et éventuellement déjà rotées)
input_dir = Path(r"C:\Users\hecto\Documents\ENS\M2\Biophysique\Corentin\rotates")

# Dossier de sortie : images recadrées
output_dir = Path(r"C:\Users\hecto\Documents\ENS\M2\Biophysique\Corentin\Crop")
output_dir.mkdir(parents=True, exist_ok=True)

# Taille du crop : largeur x hauteur
CROP_W = 2450
CROP_H = 1300
# =========================================================


def parse_timestamp_from_name(path: Path) -> datetime:
    """
    Extrait un datetime à partir du début du nom de fichier, supposé être :
    YYYYMMDD_HHMM...
    Exemple : '20241002_1616.tif', '20241002_1616_blah.tif', etc.
    """
    stem = path.stem  # nom sans extension
    key = stem[:13]   # 'YYYYMMDD_HHMM'
    return datetime.strptime(key, "%Y%m%d_%H%M")


# ========= DÉFINITION DES PLAGES TEMPORELLES + ANCRAGES =========
# À ADAPTER EN FONCTION DE TES BESOINS
#
# start : datetime ou None (None = pas de borne inférieure)
# end   : datetime ou None (None = pas de borne supérieure)
# anchor: (x_br, y_br) = coordonnée du coin inférieur droit du crop
#
# Convention ci-dessous : plage = [start, end)
# c'est-à-dire : start <= ts < end

t1 = datetime(2024, 10, 4, 10, 40)  # 20241004_1040
t2 = datetime(2024, 10, 8, 6, 29)
t3 = datetime(2024, 10, 8, 14, 22)
t4 = datetime(2024, 10, 10, 9, 42)  # EXEMPLE

# >>> ICI tu mets les coordonnées x_br, y_br que tu as mesurées
#     pour chaque plage (avec par ex. ImageJ, Fiji, etc.)
ranges = [
    {
        "name": "avant_t1",
        "start": None,
        "end": t1,
        "anchor": (2504, 1325),  # <-- À REMPLACER
    },
    {
        "name": "entre_t1_t2",
        "start": t1,
        "end": t2,
        "anchor": (2512, 1352),  # <-- À REMPLACER
    },
    {
        "name": "entre_t2_t3",
        "start": t2,
        "end": t3,
        "anchor": (2520, 1340),  # <-- À REMPLACER
    },
    {
        "name": "entre_t3_t4",
        "start": t3,
        "end": t4,
        "anchor": (2519, 1350),  # <-- À REMPLACER
    },
    {
        "name": "apres_t4",
        "start": t4,
        "end": None,
        "anchor": (2530, 1315),  # <-- À REMPLACER
    },
]
# ================================================================


def get_anchor_for_timestamp(ts: datetime):
    """Retourne (x_br, y_br) en fonction de la plage temporelle."""
    for r in ranges:
        start = r["start"]
        end = r["end"]

        cond_start = True if start is None else (ts > start)
        cond_end = True if end is None else (ts <= end)

        if cond_start and cond_end:
            return r["anchor"], r["name"]

    # Si aucune plage ne correspond (cas limite) :
    return None, None


def crop_with_bottom_right(im: Image.Image, x_br: int, y_br: int,
                           crop_w: int, crop_h: int) -> Image.Image:
    """
    Crée un crop de taille (crop_w, crop_h) tel que le point (x_br, y_br)
    soit le coin inférieur droit du rectangle.
    """
    w, h = im.size

    left = x_br - crop_w
    top = y_br - crop_h
    right = x_br
    bottom = y_br

    # Option : sécuriser contre les dépassements (si tu es sûr de tes points, ça ne devrait pas arriver)
    if left < 0 or top < 0 or right > w or bottom > h:
        raise ValueError(
            f"Crop hors limites pour image {w}x{h} avec bottom-right=({x_br},{y_br}) "
            f"et taille={crop_w}x{crop_h}. Rectangle=({left},{top},{right},{bottom})"
        )

    return im.crop((left, top, right, bottom))


# ================== BOUCLE PRINCIPALE ==================
for tif_path in sorted(input_dir.glob("*.tif")):
    ts = parse_timestamp_from_name(tif_path)
    (anchor, range_name) = get_anchor_for_timestamp(ts)

    out_path = output_dir / tif_path.name

    if anchor is None:
        # Aucune plage définie pour ce timestamp : on copie brut
        shutil.copy2(tif_path, out_path)
        print(f"[COPIE SANS CROP] {tif_path.name} (pas de plage trouvée)")
        continue

    x_br, y_br = anchor

    with Image.open(tif_path) as im:
        cropped = crop_with_bottom_right(im, x_br, y_br, CROP_W, CROP_H)
        cropped.save(out_path)

    print(f"[CROP {range_name}] {tif_path.name} -> {out_path.name} avec BR=({x_br},{y_br})")
# =======================================================


#%% PLOT EN FONCTION DU RAYONEvolution du rayon du champi au court du temps 
import matplotlib.pyplot as plt
frame = [1,10,20,30,40,50,60]
r = [500,752,1060,1350,1920,2200,2550] #(px)
plt.figure()
plt.plot(r,frame)


#%%
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ================== PARAMÈTRES ==================
input_dir = Path(r"C:\Users\hecto\Documents\ENS\M2\Biophysique\Corentin\traite3crop")
EXT = "*.tif"          # pattern des fichiers

# Coordonnées du centre (attention : x = colonne, y = ligne)
CENTER_X = 800  # à adapter : ton pixel en x (colonne)
CENTER_Y = 1290  # à adapter : ton pixel en y (ligne)

DR = 15.0               # épaisseur des anneaux en pixels
PLOT_EVERY_N = 5      # on affiche 1 plot toutes les 10 images

SAVE_PROFILES = True
profiles_output = input_dir / "radial_profiles.npz"
# =================================================


def compute_radial_map(shape, cx, cy, dr):
    """
    Pré-calcul :
    - carte des distances radiales r
    - index de bin radial pour chaque pixel (r_bins)
    - valeurs de rayon associées aux bins (r_values)

    IMPORTANT : ici on NE coupe PAS les rayons "trop grands".
    Même si le cercle est partiel dans l'image, on le garde.
    """
    H, W = shape
    y, x = np.indices((H, W))  # y = lignes, x = colonnes
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # bin radial : k = floor(r / dr)
    r_bins = (r / dr).astype(int)

    max_bin = r_bins.max()
    # Rayon "au centre" de chaque anneau
    r_values = (np.arange(max_bin + 1) + 0.5) * dr

    return r, r_bins, r_values


def compute_radial_profile(image_array, r_bins):
    """
    v_moy(r) = moyenne des pixels pour chaque bin radial.
    On prend TOUS les pixels de l'image (pas de masque "cercle complet").
    """
    vals = image_array.ravel()
    bins = r_bins.ravel()

    sums = np.bincount(bins, weights=vals)
    counts = np.bincount(bins)

    profile = sums / np.maximum(counts, 1)
    return profile


# ================== BOUCLE PRINCIPALE ==================
image_paths = sorted(input_dir.glob(EXT))
if not image_paths:
    raise FileNotFoundError(f"Aucune image trouvée dans {input_dir} avec pattern {EXT}")

radial_profiles = []
filenames = []

# Première image pour définir la taille
with Image.open(image_paths[0]) as im0:
    im0_gray = im0.convert("L")
    arr0 = np.array(im0_gray, dtype=np.float32)
    H, W = arr0.shape

# Pré-calcul carte radiale
_, r_bins, r_values = compute_radial_map(
    shape=(H, W),
    cx=CENTER_X,
    cy=CENTER_Y,
    dr=DR
)

print(f"Taille image = {W} x {H}")
print(f"Centre = ({CENTER_X}, {CENTER_Y})")
print(f"Nombre de bins radiaux = {len(r_values)}")

for idx, img_path in enumerate(image_paths):
    print(f"[{idx+1}/{len(image_paths)}] {img_path.name}")

    with Image.open(img_path) as im:
        gray = im.convert("L")
        arr = np.array(gray, dtype=np.float32)

    if arr.shape != (H, W):
        raise ValueError(
            f"L'image {img_path.name} n'a pas la même taille que la première : "
            f"{arr.shape} vs {(H, W)}"
        )

    profile = compute_radial_profile(arr, r_bins)
    radial_profiles.append(profile)
    filenames.append(img_path.name)

    # === PLOT toutes les N images ===
    if (idx % PLOT_EVERY_N) == 0 and idx<=40:
        fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(10, 4))

        # --------- Image + cercles ---------
        ax_img.imshow(arr, cmap="gray", origin="upper")
        ax_img.set_title(f"Image + cercles\n{img_path.name}")

        # on trace les cercles aux bords des anneaux
        # (0, DR, 2DR, ..., max_rayon)
        max_bin_idx = len(profile) - 1
        radius_edges = np.arange(0, (max_bin_idx + 1) * DR, DR)

        for radius in radius_edges:
            circ = Circle(
                (CENTER_X, CENTER_Y),
                radius,
                fill=False,
                linewidth=0.5
            )
            ax_img.add_patch(circ)

        # on fixe les limites pour bien voir tout
        ax_img.set_xlim(0, W - 1)
        ax_img.set_ylim(H - 1, 0)  # inversion de l'axe y (origine en haut)

        # point centre (optionnel)
        ax_img.plot(CENTER_X, CENTER_Y, 'r+', markersize=8)

        # --------- Profil radial ---------
        ax_prof.plot(r_values[:len(profile)], profile, '-')
        ax_prof.set_xlabel("Rayon r (pixels)")
        ax_prof.set_ylabel("⟨I⟩(r)")
        ax_prof.set_title(f"Profil radial\nframe {idx}")
        ax_prof.set_xlim(0,1500)

        fig.tight_layout()
        plt.show()

# Sauvegarde des profils
if SAVE_PROFILES:
    radial_profiles_arr = np.vstack(radial_profiles)
    np.savez(
        profiles_output,
        profiles=radial_profiles_arr,
        r_values=r_values,
        filenames=np.array(filenames)
    )
    print(f"Profils radiaux sauvegardés dans {profiles_output}")
    
#%% SKELETISATION

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
import networkx as nx  # optionnel si tu veux le graphe

# =============== PARAMÈTRES =================
input_dir = Path(r"C:\Users\hecto\Documents\ENS\M2\Biophysique\Corentin\traite2")
output_dir = Path(r"C:\Users\hecto\Documents\ENS\M2\Biophysique\Corentin\skeleton")
output_dir.mkdir(parents=True, exist_ok=True)

PATTERN = "*.tif"   # ou "*.png" suivant ton cas

# pour l’overlay
SHOW_PLOTS = False      # passe à False si tu veux juste sauver les png
SAVE_OVERLAY = True

REMOVE_PURE_ENDPOINT_COMPONENTS = True  # vire les composants sans jonction

# ============================================


def img_to_binary(arr):
    """Gris -> image binaire (True = arbre)."""
    arr = arr.astype(np.float32)
    thr = threshold_otsu(arr)
    # ton arbre est noir sur fond blanc => arbre = valeurs < thr
    binary = arr < thr
    return binary


def skeleton_and_nodes(binary):
    """
    binary : booléen (True = arbre)
    retourne :
        skel      : squelette booléen
        endpoints : masque booléen (1 voisin)
        junctions : masque booléen (>=3 voisins)
    """
    # squelette
    skel = skeletonize(binary)

    # comptage des voisins (8-connexité)
    skel_uint = skel.astype(np.uint8)
    from scipy.ndimage import convolve

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbor_count = convolve(skel_uint, kernel, mode="constant", cval=0)

    endpoints = (skel & (neighbor_count == 1))
    junctions = (skel & (neighbor_count >= 3))

    return skel, endpoints, junctions, neighbor_count


def build_graph_from_skeleton(skel, endpoints, junctions):
    """
    Construit un graphe (networkx) où:
      - chaque endpoint / junction = noeud
      - chaque chaîne de pixels entre 2 noeuds = arête
    """
    G = nx.Graph()

    H, W = skel.shape
    skel_idx = np.argwhere(skel)

    # index des noeuds
    node_mask = endpoints | junctions
    node_coords = np.argwhere(node_mask)

    # mapping coord -> id
    coord_to_id = {tuple(c): i for i, c in enumerate(map(tuple, node_coords))}
    for i, (y, x) in enumerate(node_coords):
        G.add_node(i, y=int(y), x=int(x))

    skel_set = set(map(tuple, skel_idx))
    node_set = set(map(tuple, node_coords))

    # petit helper pour voisins
    def neighbors(y, x):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx_ < W:
                    if (ny, nx_) in skel_set:
                        yield ny, nx_

    visited = set()

    # pour chaque noeud, on lance des marches le long du squelette
    for yc, xc in node_coords:
        start = (int(yc), int(xc))
        start_id = coord_to_id[start]

        for ny, nx_ in neighbors(*start):
            if (start, (ny, nx_)) in visited or ((ny, nx_), start) in visited:
                continue

            path = [start, (ny, nx_)]
            prev = start
            current = (ny, nx_)

            while True:
                if current in node_set and current != start:
                    # on a atteint un autre noeud
                    end_id = coord_to_id[current]
                    G.add_edge(start_id, end_id, path=path[:])
                    break

                # sinon on continue à suivre le pixel suivant
                neigh = [p for p in neighbors(*current) if p != prev]

                if len(neigh) == 0:
                    # extrémité qui n'est pas marquée comme endpoint
                    break
                if len(neigh) > 1:
                    # on arrive à une jonction (qui devrait être dans node_set)
                    # mais au cas où, on arrête là
                    break

                nxt = neigh[0]
                visited.add((current, nxt))
                path.append(nxt)
                prev, current = current, nxt

    return G


def overlay_and_save(original, skel, endpoints, junctions, out_path):
    """Crée une image RGB avec squelette + noeuds et sauvegarde."""
    H, W = original.shape
    # base = normalisée en [0,1] pour affichage
    base = (original - original.min()) / max(1e-6, (original.max() - original.min()))
    rgb = np.dstack([base, base, base])

    # squelette en noir
    rgb[skel] = [0, 0, 0]
    # endpoints en bleu
    rgb[endpoints] = [0, 0, 1]
    # junctions en rouge
    rgb[junctions] = [1, 0, 0]

    plt.figure(figsize=(8, 6))
    plt.imshow(rgb, origin="upper")
    plt.axis("off")
    plt.title(out_path.name)

    if SAVE_OVERLAY:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
        
from scipy.ndimage import label  # déjà scipy.ndimage.convolve, donc cohérent

def remove_components_without_junctions(skel, endpoints, junctions):
    """
    Ne garde que les composantes du squelette qui contiennent
    au moins UNE jonction (pixel avec neighbor_count >= 3).

    Tout composant qui n'a que des extrémités (et du milieu de segment)
    est supprimé : utile pour virer les petits grains de poussière.
    """
    labeled, num = label(skel, structure=np.ones((3, 3), dtype=int))

    if num == 0:
        return skel, endpoints, junctions

    keep_mask = np.zeros_like(skel, dtype=bool)

    for lab_id in range(1, num + 1):
        comp_mask = (labeled == lab_id)

        # est-ce qu'il y a AU MOINS UNE jonction dans ce composant ?
        if (junctions & comp_mask).any():
            keep_mask |= comp_mask  # on garde tout ce composant

    # on applique le masque de conservation
    skel_clean = skel & keep_mask
    endpoints_clean = endpoints & keep_mask
    junctions_clean = junctions & keep_mask

    return skel_clean, endpoints_clean, junctions_clean



# ================== MAIN =====================
files = sorted(input_dir.glob(PATTERN))
if not files:
    raise FileNotFoundError(f"Aucune image trouvée dans {input_dir}")

all_profiles = []  # si tu veux plus tard analyser les noeuds / longueurs etc.

for f in files:
    print(f"[TRAITEMENT] {f.name}")
    with Image.open(f) as im:
        gray = im.convert("L")
        arr = np.array(gray, dtype=np.float32)

    binary = img_to_binary(arr)
    skel, endpoints, junctions, neighbor_count = skeleton_and_nodes(binary)
    
    if REMOVE_PURE_ENDPOINT_COMPONENTS:
        skel, endpoints, junctions = remove_components_without_junctions(
            skel, endpoints, junctions
        )
    
    # optionnel : graphe (maintenant nettoyé)
    G = build_graph_from_skeleton(skel, endpoints, junctions)


    # overlay visuel
    out_png = output_dir / (f.stem + "_skeleton_overlay.png")
    overlay_and_save(arr, skel, endpoints, junctions, out_png)

