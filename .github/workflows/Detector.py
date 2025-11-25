import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import morphology, filters
from skimage.morphology import skeletonize
from scipy import ndimage
import pandas as pd

class FungalNetworkDetector:
    """
    Classe pour détecter et analyser les réseaux mycéliens dans des séquences d'images
    """
    
    def __init__(self, min_filament_width=1, max_filament_width=10):
        """
        Args:
            min_filament_width: Largeur minimale des filaments (pixels)
            max_filament_width: Largeur maximale des filaments (pixels)
        """
        self.min_width = min_filament_width
        self.max_width = max_filament_width

    def auto_invert_image(self, img):
        """
        Déterminer automatiquement si l'image doit être inversée
        (champignon sombre sur fond clair -> inverser)
        """
        # Calculer la moyenne de l'intensité
        mean_intensity = np.mean(img)
        
        # Si la moyenne est élevée (>127), le fond est clair
        # Donc les structures sont sombres -> inverser
        if mean_intensity > 127:
            print("Inversion de l'image (structures sombres sur fond clair)")
            return cv2.bitwise_not(img)
        else:
            print(" Image correcte (structures claires sur fond sombre)")
        return img
    
    def filter_by_shape(self, binary, min_aspect_ratio=2.0, min_elongation=0.3):
        """
        Filtrer les objets qui ne ressemblent pas à des filaments
        
        Args:
            min_aspect_ratio: Ratio longueur/largeur minimum pour un filament
            min_elongation: Score d'élongation minimum (0-1)
        """
        # Trouver tous les contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Créer un masque vide
        filtered = np.zeros_like(binary)
        
        print(f"  Analyse de {len(contours)} objets détectés...")
        kept = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Ignorer les très petits objets
            if area < 20:
                continue
            
            # Calculer le rectangle englobant orienté
            if len(contour) >= 5:  # Besoin d'au moins 5 points
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (center, (width, height), angle) = ellipse
                    
                    # Ratio d'aspect (le plus grand divisé par le plus petit)
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                    else:
                        aspect_ratio = 1.0
                    
                    # Les filaments ont un grand ratio d'aspect
                    if aspect_ratio >= min_aspect_ratio:
                        cv2.drawContours(filtered, [contour], -1, 255, -1)
                        kept += 1
                except:
                    # Si fitEllipse échoue, utiliser le rectangle englobant simple
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 0 and h > 0:
                        aspect_ratio = max(w, h) / min(w, h)
                        if aspect_ratio >= min_aspect_ratio:
                            cv2.drawContours(filtered, [contour], -1, 255, -1)
                            kept += 1
            else:
                # Pour les petits contours, garder seulement s'ils sont très allongés
                x, y, w, h = cv2.boundingRect(contour)
                if w > 0 and h > 0:
                    aspect_ratio = max(w, h) / min(w, h)
                    if aspect_ratio >= min_aspect_ratio * 1.5:  # Critère plus strict
                        cv2.drawContours(filtered, [contour], -1, 255, -1)
                        kept += 1
        
        print(f"{kept}/{len(contours)} objets conservés (filaments)")
        return filtered

    def filter_isolated_pixels(self, binary, min_neighbors=3):
        """
        Enlever les pixels isolés (bruit) en comptant leurs voisins connectés
        """
        # Créer un kernel pour compter les voisins
        kernel = np.ones((3, 3), np.uint8)
        
        # Dilater légèrement puis éroder pour connecter les structures proches
        temp = cv2.dilate(binary, kernel, iterations=1)
        temp = cv2.erode(temp, kernel, iterations=1)
        
        # Compter les voisins pour chaque pixel
        neighbor_count = cv2.filter2D(binary // 255, -1, kernel)
        
        # Garder seulement les pixels avec suffisamment de voisins
        filtered = np.where(neighbor_count >= min_neighbors, binary, 0).astype(np.uint8)
        
        return filtered

    def remove_border_noise(self, binary, border_width=50):
        """
        Enlever le bruit sur les bords de l'image (souvent des artefacts)
        
        Args:
            border_width: Largeur de la bordure à nettoyer (pixels)
        """
        h, w = binary.shape
        mask = np.ones_like(binary)
        
        # Créer un masque qui exclut les bords
        mask[:border_width, :] = 0  # Haut
        mask[-border_width:, :] = 0  # Bas
        mask[:, :border_width] = 0  # Gauche
        mask[:, -border_width:] = 0  # Droite
        
        # Appliquer le masque
        filtered = cv2.bitwise_and(binary, binary, mask=mask)
        
        return filtered

    def keep_largest_components(self, binary, n_components=5):
        """
        Garder seulement les N plus grandes composantes connexes
        (utile si le réseau mycélien est la plus grande structure)
        """
        # Trouver toutes les composantes connexes
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Trier par taille (aire) décroissante, en excluant le fond (label 0)
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)
        
        # Garder seulement les N plus grandes
        filtered = np.zeros_like(binary)
        
        for i in range(min(n_components, len(areas))):
            label_id = areas[i][0]
            area_size = areas[i][1]
            filtered[labels == label_id] = 255
            print(f"  Composante {i+1}: {area_size} pixels")
        
        return filtered

    def advanced_cleaning(self, binary, config='balanced'):
        """
        Pipeline de nettoyage avancé avec différentes configurations
        
        Args:
            config: 'aggressive' (enlève beaucoup), 'balanced', 'gentle' (conservateur)
        """
        configs = {
            'aggressive': {
                'min_object_size': 200,
                'min_aspect_ratio': 3.0,
                'min_neighbors': 4,
                'border_width': 80,
                'n_components': 3,
                'closing_size': 2
            },
            'balanced': {
                'min_object_size': 100,
                'min_aspect_ratio': 2.5,
                'min_neighbors': 3,
                'border_width': 50,
                'n_components': 5,
                'closing_size': 3
            },
            'gentle': {
                'min_object_size': 50,
                'min_aspect_ratio': 2.0,
                'min_neighbors': 2,
                'border_width': 30,
                'n_components': 10,
                'closing_size': 4
            }
        }
        
        params = configs.get(config, configs['balanced'])
        
        print(f"Nettoyage avancé (mode: {config})")
        
        # 1. Enlever le bruit sur les bords
        cleaned = self.remove_border_noise(binary, params['border_width'])
        
        # 2. Fermeture morphologique pour connecter les fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                        (params['closing_size'], params['closing_size']))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # 3. Enlever les petits objets
        cleaned = morphology.remove_small_objects(
            cleaned.astype(bool),
            min_size=params['min_object_size']
        ).astype(np.uint8) * 255
        
        # 4. Filtrer par forme (garder les filaments)
        cleaned = self.filter_by_shape(cleaned, 
                                    min_aspect_ratio=params['min_aspect_ratio'])
        
        # 5. Enlever les pixels isolés
        cleaned = self.filter_isolated_pixels(cleaned, 
                                            min_neighbors=params['min_neighbors'])
        
        # 6. Garder seulement les plus grandes composantes
        cleaned = self.keep_largest_components(cleaned, 
                                            n_components=params['n_components'])
        
        return cleaned
    
    def validate_detection(self, binary, skeleton, min_length=100, max_density=0.5):
        """
        Valider que la détection correspond bien à un réseau mycélien
        
        Args:
            binary: Masque binaire du réseau
            skeleton: Squelette du réseau
            min_length: Longueur minimale du réseau (pixels)
            max_density: Densité maximale acceptable (0-1)
        
        Returns:
            bool: True si la détection est valide
        """
        # Calculer les métriques de base
        total_area = np.sum(binary > 0)
        total_length = np.sum(skeleton > 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        density = total_area / total_pixels
        
        # Critères de validation
        valid = True
        
        if total_length < min_length:
            print(f"  ⚠️ Réseau trop court: {total_length} < {min_length}")
            valid = False
        
        if density > max_density:
            print(f"  ⚠️ Densité trop élevée: {density:.3f} > {max_density} (probablement du bruit)")
            valid = False
        
        if total_area == 0:
            print(f"  ⚠️ Aucun réseau détecté")
            valid = False
        
        if valid:
            print(f"  ✓ Détection valide (longueur: {total_length}, densité: {density:.3f})")
        
        return valid

    def preprocess_image(self, image_path, enhance_contrast=True, denoise=True, denoise_strength=7, auto_invert=True):
        """
        Prétraitement de l'image avec débruitage avancé
        
        Args:
            enhance_contrast: Améliorer le contraste
            denoise: Appliquer un débruitage
            denoise_strength: Force du débruitage (5-15 recommandé)
        """
        # Charger l'image
        if isinstance(image_path, (str, Path)):
            image_path = str(image_path)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
        else:
            img = image_path
        
        # Vérifications
        if not isinstance(img, np.ndarray):
            raise ValueError(f"L'image doit être un numpy array, type reçu: {type(img)}")
        
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # === INVERSION AUTOMATIQUE ===
        if auto_invert:
            img = self.auto_invert_image(img)

        # === DÉBRUITAGE AVANCÉ ===
        if denoise:
            # 1. Filtre bilatéral (préserve les bords)
            img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            
            # 2. Débruitage non-local means (très efficace)
            img = cv2.fastNlMeansDenoising(img, None, h=denoise_strength, templateWindowSize=7, searchWindowSize=21)
        
        # === AMÉLIORATION DU CONTRASTE ===
        if enhance_contrast:
            # CLAHE avec paramètres optimisés pour les réseaux mycéliens
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img = clahe.apply(img)
        
        return img
            
        if img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Égalisation d'histogramme adaptative (CLAHE)
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
        
        return img
    
    def detect_network_threshold(self, img, method='adaptive'):
        """
        Détection par seuillage
        
        Args:
            method: 'adaptive', 'otsu', 'triangle', ou 'multiotsu'
        """
        if method == 'adaptive':
            # Seuillage adaptatif - bon pour l'illumination non-uniforme
            binary = cv2.adaptiveThreshold(
                img, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                blockSize=15, 
                C=5
            )
        elif method == 'otsu':
            # Méthode d'Otsu
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif method == 'triangle':
            # Méthode du triangle
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        elif method == 'multiotsu':
            # Multi-Otsu pour séparer différentes intensités
            from skimage.filters import threshold_multiotsu
            thresholds = threshold_multiotsu(img, classes=3)
            binary = (img < thresholds[0]).astype(np.uint8) * 255
        
        return binary
    
    def detect_network_frangi(self, img, scales=range(1, 8, 2), alpha=0.5, beta=0.5, gamma=15):
        """
        Détection par filtre de Frangi (détection de structures tubulaires)
        Excellent pour les réseaux filamenteux
        """
        from skimage.filters import frangi
        
        # Normaliser l'image
        img_normalized = img.astype(float) / 255.0
        
        # Appliquer le filtre de Frangi
        frangi_response = frangi(
            img_normalized, 
            sigmas=scales,
            alpha=alpha, 
            beta=beta, 
            gamma=gamma,
            black_ridges=True
        )
        
        # Convertir en image binaire
        threshold = filters.threshold_otsu(frangi_response)
        binary = (frangi_response > threshold).astype(np.uint8) * 255
        
        return binary, frangi_response
    
    def detect_network_hessian(self, img, scales=range(1, 8, 2)):
        """
        Détection par filtre Hessian (alternative au Frangi)
        """
        from skimage.filters import hessian
        
        img_normalized = img.astype(float) / 255.0
        
        hessian_response = hessian(img_normalized, sigmas=scales, black_ridges=True)
        
        threshold = filters.threshold_otsu(hessian_response)
        binary = (hessian_response > threshold).astype(np.uint8) * 255
        
        return binary, hessian_response
    
    def clean_binary_mask(self, binary, min_size=50, closing_size=3):
        """
        Nettoyer le masque binaire (enlever le bruit, combler les trous)
        """
        # Fermeture morphologique pour connecter les structures proches
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Ouverture pour enlever les petits bruits
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
        
        # Enlever les petites composantes
        cleaned = morphology.remove_small_objects(
            cleaned.astype(bool), 
            min_size=min_size
        ).astype(np.uint8) * 255
        
        return cleaned
    
    def skeletonize_network(self, binary):
        """
        Squelettisation du réseau pour obtenir la topologie
        """
        # Conversion en booléen
        binary_bool = binary.astype(bool)
        
        # Squelettisation
        skeleton = skeletonize(binary_bool)
        
        return skeleton.astype(np.uint8) * 255
    
    def analyze_network_metrics(self, binary, skeleton):
        """
        Calculer les métriques du réseau
        """
        metrics = {}
        
        # Surface totale du réseau
        metrics['total_area'] = np.sum(binary > 0)
        
        # Longueur totale (approximation par le squelette)
        metrics['total_length'] = np.sum(skeleton > 0)
        
        # Densité du réseau
        total_pixels = binary.shape[0] * binary.shape[1]
        metrics['network_density'] = metrics['total_area'] / total_pixels
        
        # Nombre de composantes connexes
        num_labels, labels = cv2.connectedComponents(binary)
        metrics['num_components'] = num_labels - 1  # -1 pour exclure le fond
        
        # Points de branchement et terminaisons
        branch_points, end_points = self._find_critical_points(skeleton)
        metrics['num_branches'] = len(branch_points)
        metrics['num_endpoints'] = len(end_points)
        
        # Largeur moyenne des filaments
        if metrics['total_length'] > 0:
            metrics['average_width'] = metrics['total_area'] / metrics['total_length']
        else:
            metrics['average_width'] = 0
        
        return metrics
    
    def _find_critical_points(self, skeleton):
        """
        Trouver les points de branchement et les terminaisons
        """
        # Créer un kernel pour compter les voisins
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # Convoluer pour compter les voisins
        skeleton_bin = (skeleton > 0).astype(np.uint8)
        neighbor_count = cv2.filter2D(skeleton_bin, -1, kernel)
        
        # Points de branchement: plus de 3 voisins (incluant le pixel central)
        branch_points = np.argwhere((neighbor_count >= 13) & (skeleton_bin == 1))
        
        # Points terminaux: exactement 1 voisin (11 = 10 + 1)
        end_points = np.argwhere((neighbor_count == 11) & (skeleton_bin == 1))
        
        return branch_points, end_points
    
    def visualize_results(self, original, binary, skeleton, metrics=None, save_path=None):
        """
        Visualiser les résultats
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Image originale
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Image originale')
        axes[0].axis('off')
        
        # Masque binaire
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title('Réseau détecté')
        axes[1].axis('off')
        
        # Squelette superposé
        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        overlay[skeleton > 0] = [255, 0, 0]  # Rouge pour le squelette
        axes[2].imshow(overlay)
        axes[2].set_title('Squelette du réseau')
        axes[2].axis('off')
        
        if metrics:
            # Ajouter les métriques comme texte
            metrics_text = '\n'.join([f'{k}: {v:.2f}' if isinstance(v, float) else f'{k}: {v}' 
                                     for k, v in metrics.items()])
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def process_single_image(self, image_path, method='frangi', visualize=True, save_path=None,
                        denoise=True, denoise_strength=7, 
                        threshold_multiplier=1.0,
                        cleaning_mode='aggressive',
                        validate=True):
        """
        Args:
            cleaning_mode: 'aggressive', 'balanced', 'gentle' ou None (désactiver)
        """
        print(f"  Chargement de l'image: {image_path}")
        
        # Prétraitement avec débruitage et inversion
        img = self.preprocess_image(image_path, enhance_contrast=True, 
                                    denoise=denoise, denoise_strength=denoise_strength,
                                    auto_invert=True)
        print(f"  Image prétraitée: {img.shape}, dtype: {img.dtype}")
        
        # Détection
        print(f"  Détection avec méthode: {method}")
        if method == 'frangi':
            binary, _ = self.detect_network_frangi(
                img, 
                threshold_multiplier=threshold_multiplier,
                black_ridges=False  # Après inversion
            )
        elif method == 'hessian':
            binary, _ = self.detect_network_hessian(img)
        else:
            binary = self.detect_network_threshold(img, method=method)
        
        # Nettoyage avancé
        if cleaning_mode:
            print(f"  Nettoyage avancé...")
            binary = self.advanced_cleaning(binary, config=cleaning_mode)
        else:
            # Nettoyage basique seulement
            print(f"  Nettoyage basique")
            binary = self.clean_binary_mask(binary, min_size=100, closing_size=3)
        
        # Squelettisation
        print(f"  Squelettisation")
        skeleton = self.skeletonize_network(binary)
        
        # Validation
        if validate:
            is_valid = self.validate_detection(binary, skeleton, 
                                            min_length=100, max_density=0.3)
            if not is_valid:
                print(f"  ⚠️ Détection invalide - résultat probablement bruité")
        
        # Analyse
        print(f"  Analyse des métriques")
        metrics = self.analyze_network_metrics(binary, skeleton)
        
        # Visualisation
        if visualize:
            self.visualize_results(img, binary, skeleton, metrics, save_path)
        
        return {
            'original': img,
            'binary': binary,
            'skeleton': skeleton,
            'metrics': metrics
        }
    
    def process_image_sequence(self, image_folder, file_pattern='*.jpg', method='frangi', 
                               output_folder=None, save_video=False, fps=5):
        """
        Traiter une séquence d'images
        """
        # Trouver toutes les images
        image_folder = Path(image_folder)
        image_files = sorted(image_folder.glob(file_pattern))
        
        if len(image_files) == 0:
            raise ValueError(f"Aucune image trouvée avec le pattern {file_pattern} dans {image_folder}")
        
        print(f"Traitement de {len(image_files)} images...")
        
        # Créer le dossier de sortie
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(exist_ok=True, parents=True)
        
        # Traiter chaque image
        results = []
        all_metrics = []
        
        for i, img_path in enumerate(image_files):
            print(f"Traitement de {img_path.name} ({i+1}/{len(image_files)})...")
            
            result = self.process_single_image(
                img_path, 
                method=method, 
                visualize=False
            )
            
            # Ajouter le timestamp/frame number
            result['metrics']['frame'] = i
            result['metrics']['filename'] = img_path.name
            
            results.append(result)
            all_metrics.append(result['metrics'])
            
            # Sauvegarder les résultats individuels
            if output_folder:
                # Sauvegarder le masque binaire
                cv2.imwrite(
                    str(output_folder / f"{img_path.stem}_binary.png"),
                    result['binary']
                )
                # Sauvegarder le squelette
                cv2.imwrite(
                    str(output_folder / f"{img_path.stem}_skeleton.png"),
                    result['skeleton']
                )
        
        # Créer un DataFrame avec toutes les métriques
        metrics_df = pd.DataFrame(all_metrics)
        
        if output_folder:
            metrics_df.to_csv(output_folder / 'network_metrics.csv', index=False)
        
        # Créer une vidéo si demandé
        if save_video and output_folder:
            self._create_video(results, output_folder / 'network_evolution.avi', fps)
        
        # Tracer l'évolution des métriques
        if output_folder:
            self._plot_metrics_evolution(metrics_df, output_folder / 'metrics_evolution.png')
        
        print(f"\n Traitement terminé!")
        return results, metrics_df
    
    def _create_video(self, results, output_path, fps=5):
        """
        Créer une vidéo montrant l'évolution du réseau
        """
        if len(results) == 0:
            return
        
        # Obtenir les dimensions
        height, width = results[0]['original'].shape
        
        # Créer le writer vidéo
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width*3, height))
        
        for result in results:
            # Créer une image avec 3 vues côte à côte
            original_rgb = cv2.cvtColor(result['original'], cv2.COLOR_GRAY2RGB)
            binary_rgb = cv2.cvtColor(result['binary'], cv2.COLOR_GRAY2RGB)
            
            # Squelette superposé
            overlay = original_rgb.copy()
            overlay[result['skeleton'] > 0] = [0, 0, 255]
            
            # Combiner les 3 vues
            combined = np.hstack([original_rgb, binary_rgb, overlay])
            
            out.write(combined)
        
        out.release()
        print(f"Vidéo sauvegardée: {output_path}")
    
    def _plot_metrics_evolution(self, metrics_df, output_path):
        """
        Tracer l'évolution des métriques au cours du temps
        """
        import seaborn as sns
        
        # Configuration du style
        sns.set_style("whitegrid")
        
        # Créer la figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Évolution des métriques du réseau mycélien', fontsize=16, fontweight='bold')
        
        # 1. Surface totale
        axes[0, 0].plot(metrics_df['frame'], metrics_df['total_area'], 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].fill_between(metrics_df['frame'], metrics_df['total_area'], alpha=0.3)
        axes[0, 0].set_xlabel('Frame', fontsize=10)
        axes[0, 0].set_ylabel('Surface (pixels)', fontsize=10)
        axes[0, 0].set_title('Surface du réseau', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Longueur totale
        axes[0, 1].plot(metrics_df['frame'], metrics_df['total_length'], 'g-', linewidth=2, marker='s', markersize=4)
        axes[0, 1].fill_between(metrics_df['frame'], metrics_df['total_length'], alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Frame', fontsize=10)
        axes[0, 1].set_ylabel('Longueur (pixels)', fontsize=10)
        axes[0, 1].set_title('Longueur du réseau', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Densité
        axes[0, 2].plot(metrics_df['frame'], metrics_df['network_density'] * 100, 'r-', linewidth=2, marker='^', markersize=4)
        axes[0, 2].fill_between(metrics_df['frame'], metrics_df['network_density'] * 100, alpha=0.3, color='red')
        axes[0, 2].set_xlabel('Frame', fontsize=10)
        axes[0, 2].set_ylabel('Densité (%)', fontsize=10)
        axes[0, 2].set_title('Densité du réseau', fontsize=12, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Points de branchement
        axes[1, 0].plot(metrics_df['frame'], metrics_df['num_branches'], 'purple', linewidth=2, marker='d', markersize=4, label='Branches')
        axes[1, 0].plot(metrics_df['frame'], metrics_df['num_endpoints'], 'orange', linewidth=2, marker='v', markersize=4, label='Terminaisons')
        axes[1, 0].set_xlabel('Frame', fontsize=10)
        axes[1, 0].set_ylabel('Nombre', fontsize=10)
        axes[1, 0].set_title('Topologie du réseau', fontsize=12, fontweight='bold')
        axes[1, 0].legend(loc='best')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Largeur moyenne
        axes[1, 1].plot(metrics_df['frame'], metrics_df['average_width'], 'brown', linewidth=2, marker='p', markersize=4)
        axes[1, 1].fill_between(metrics_df['frame'], metrics_df['average_width'], alpha=0.3, color='brown')
        axes[1, 1].set_xlabel('Frame', fontsize=10)
        axes[1, 1].set_ylabel('Largeur (pixels)', fontsize=10)
        axes[1, 1].set_title('Largeur moyenne des filaments', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Nombre de composantes
        axes[1, 2].plot(metrics_df['frame'], metrics_df['num_components'], 'teal', linewidth=2, marker='h', markersize=4)
        axes[1, 2].fill_between(metrics_df['frame'], metrics_df['num_components'], alpha=0.3, color='teal')
        axes[1, 2].set_xlabel('Frame', fontsize=10)
        axes[1, 2].set_ylabel('Nombre', fontsize=10)
        axes[1, 2].set_title('Composantes connexes', fontsize=12, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graphique d'évolution sauvegardé: {output_path}")
        