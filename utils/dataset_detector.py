"""
Universal Dataset Detector and Converter
Supports: LFW, YaleB, CelebA, VGGFace2, CASIA-WebFace, MegaFace
"""
import os
import zipfile
import glob
from pathlib import Path
from .sse import sse_print

class DatasetDetector:
    """
    Detect and convert various face recognition datasets
    """
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.dataset_type = None
        self.dataset_info = {}
        
    def detect_dataset_type(self):
        """
        Detect which dataset is being used
        Returns: dataset_type (str)
        """
        sse_print("dataset_detection", {}, progress=21,
                 message="检测数据集类型...",
                 log="[21%] 正在检测数据集类型\n")
        
        # Check for specific dataset markers
        if self._is_lfw():
            self.dataset_type = "LFW"
        elif self._is_yaleb():
            self.dataset_type = "YaleB"
        elif self._is_celeba():
            self.dataset_type = "CelebA"
        elif self._is_vggface2():
            self.dataset_type = "VGGFace2"
        elif self._is_casia_webface():
            self.dataset_type = "CASIA-WebFace"
        elif self._is_megaface():
            self.dataset_type = "MegaFace"
        else:
            self.dataset_type = "Generic"
        
        sse_print("dataset_detected", {}, progress=21,
                 message=f"检测到数据集类型: {self.dataset_type}",
                 log=f"[21%] 数据集类型: {self.dataset_type}\n",
                 details={"dataset_type": self.dataset_type})
        
        return self.dataset_type
    
    def _is_lfw(self):
        """Check if this is LFW dataset"""
        # LFW structure: lfw/PersonName/*.jpg
        if (self.data_path / 'lfw').exists() or \
           any('lfw' in str(p).lower() for p in self.data_path.glob('*')):
            return True
        # Check for lfw.zip
        if list(self.data_path.glob('*lfw*.zip')):
            return True
        return False
    
    def _is_yaleb(self):
        """Check if this is YaleB dataset"""
        if list(self.data_path.glob('*yale*.zip')) or \
           list(self.data_path.glob('*耶鲁*.zip')):
            return True
        if any('yale' in str(p).lower() for p in self.data_path.glob('*')):
            return True
        return False
    
    def _is_celeba(self):
        """Check if this is CelebA dataset"""
        if list(self.data_path.glob('*celeba*.zip')) or \
           list(self.data_path.glob('*celeba*.7z')):
            return True
        if (self.data_path / 'Anno').exists() or (self.data_path / 'Img').exists():
            return True
        return False
    
    def _is_vggface2(self):
        """Check if this is VGGFace2 dataset"""
        if list(self.data_path.glob('*vggface*.zip')) or \
           'vggface' in str(self.data_path).lower():
            return True
        if (self.data_path / 'meta').exists() and (self.data_path / 'data').exists():
            return True
        return False
    
    def _is_casia_webface(self):
        """Check if this is CASIA-WebFace dataset"""
        if 'casia' in str(self.data_path).lower() or \
           'webface' in str(self.data_path).lower():
            return True
        return False
    
    def _is_megaface(self):
        """Check if this is MegaFace dataset"""
        if 'megaface' in str(self.data_path).lower():
            return True
        return False
    
    def prepare_dataset(self, max_samples=50):
        """
        Prepare dataset for use (extract if needed, convert format)
        Returns: (prepared_path, image_list)
        """
        dataset_type = self.detect_dataset_type()
        
        if dataset_type == "LFW":
            return self._prepare_lfw(max_samples)
        elif dataset_type == "YaleB":
            return self._prepare_yaleb(max_samples)
        elif dataset_type == "CelebA":
            return self._prepare_celeba(max_samples)
        elif dataset_type == "VGGFace2":
            return self._prepare_vggface2(max_samples)
        elif dataset_type == "CASIA-WebFace":
            return self._prepare_casia(max_samples)
        elif dataset_type == "MegaFace":
            return self._prepare_megaface(max_samples)
        else:
            return self._prepare_generic(max_samples)
    
    def _prepare_lfw(self, max_samples):
        """Prepare LFW dataset"""
        sse_print("preparing_lfw", {}, progress=22,
                 message="准备LFW数据集...",
                 log="[22%] 准备LFW数据集\n")
        
        # Check for lfw.zip
        zip_files = list(self.data_path.glob('*lfw*.zip'))
        if zip_files:
            extract_dir = self.data_path / '.extracted' / 'lfw'
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            # Check cache
            existing = list(extract_dir.rglob('*.jpg'))
            if len(existing) >= 10:
                sse_print("using_cached_lfw", {}, progress=23,
                         message=f"使用缓存的LFW数据: {len(existing)}张",
                         log=f"[23%] LFW缓存: {len(existing)}张\n")
                return str(extract_dir), existing[:max_samples]
            
            # Extract
            try:
                with zipfile.ZipFile(zip_files[0], 'r') as zf:
                    members = [m for m in zf.namelist() if m.endswith('.jpg')][:max_samples * 2]
                    for m in members:
                        try: zf.extract(m, extract_dir)
                        except: pass
                extracted = list(extract_dir.rglob('*.jpg'))
                if extracted:
                    sse_print("lfw_extracted", {}, progress=24,
                             message=f"LFW数据集已提取: {len(extracted)}张",
                             log=f"[24%] LFW提取完成\n")
                    return str(extract_dir), extracted[:max_samples]
            except: pass
        
        # Check for existing lfw directory
        lfw_dir = self.data_path / 'lfw'
        if not lfw_dir.exists():
            for subdir in self.data_path.glob('*'):
                if 'lfw' in str(subdir).lower() and subdir.is_dir():
                    lfw_dir = subdir
                    break
        
        if lfw_dir.exists():
            images = list(lfw_dir.rglob('*.jpg'))[:max_samples]
            if images:
                sse_print("lfw_ready", {}, progress=24,
                         message=f"LFW数据集就绪: {len(images)}张",
                         log=f"[24%] LFW: {len(images)}张图片\n")
                return str(lfw_dir), images
        
        return None, []
    
    def _prepare_yaleb(self, max_samples):
        """Prepare YaleB dataset"""
        sse_print("preparing_yaleb", {}, progress=22,
                 message="准备YaleB数据集...",
                 log="[22%] 准备YaleB数据集\n")
        
        # Check for yale zip
        zip_files = list(self.data_path.glob('*yale*.zip')) + list(self.data_path.glob('*耶鲁*.zip'))
        if zip_files:
            extract_dir = self.data_path / '.extracted' / 'yaleb'
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            existing = list(extract_dir.rglob('*.pgm')) + list(extract_dir.rglob('*.jpg'))
            if len(existing) >= 10:
                return str(extract_dir), existing[:max_samples]
            
            try:
                with zipfile.ZipFile(zip_files[0], 'r') as zf:
                    members = [m for m in zf.namelist() 
                             if m.endswith(('.pgm', '.jpg'))][:max_samples * 2]
                    for m in members:
                        try: zf.extract(m, extract_dir)
                        except: pass
                extracted = list(extract_dir.rglob('*.pgm')) + list(extract_dir.rglob('*.jpg'))
                if extracted:
                    sse_print("yaleb_extracted", {}, progress=24,
                             message=f"YaleB数据集已提取: {len(extracted)}张",
                             log=f"[24%] YaleB提取完成\n")
                    return str(extract_dir), extracted[:max_samples]
            except: pass
        
        return None, []
    
    def _prepare_celeba(self, max_samples):
        """Prepare CelebA dataset"""
        sse_print("preparing_celeba", {}, progress=22,
                 message="准备CelebA数据集...",
                 log="[22%] 准备CelebA数据集\n")
        
        # Check for existing Img directory
        img_dir = self.data_path / 'Img'
        if img_dir.exists():
            images = list(img_dir.rglob('*.jpg'))[:max_samples]
            if images:
                sse_print("celeba_ready", {}, progress=24,
                         message=f"CelebA数据集就绪: {len(images)}张",
                         log=f"[24%] CelebA: {len(images)}张\n")
                return str(img_dir), images
        
        # Check for zip files
        zip_files = list(self.data_path.glob('*celeba*.zip'))
        if zip_files:
            extract_dir = self.data_path / '.extracted' / 'celeba'
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            existing = list(extract_dir.rglob('*.jpg'))
            if len(existing) >= 10:
                return str(extract_dir), existing[:max_samples]
            
            try:
                with zipfile.ZipFile(zip_files[0], 'r') as zf:
                    members = [m for m in zf.namelist() if m.endswith('.jpg')][:max_samples * 2]
                    for m in members:
                        try: zf.extract(m, extract_dir)
                        except: pass
                extracted = list(extract_dir.rglob('*.jpg'))
                if extracted:
                    return str(extract_dir), extracted[:max_samples]
            except: pass
        
        return None, []
    
    def _prepare_vggface2(self, max_samples):
        """Prepare VGGFace2 dataset"""
        sse_print("preparing_vggface2", {}, progress=22,
                 message="准备VGGFace2数据集...",
                 log="[22%] 准备VGGFace2数据集\n")
        
        data_dir = self.data_path / 'data'
        if data_dir.exists():
            images = list(data_dir.rglob('*.jpg'))[:max_samples]
            if images:
                sse_print("vggface2_ready", {}, progress=24,
                         message=f"VGGFace2数据集就绪: {len(images)}张",
                         log=f"[24%] VGGFace2: {len(images)}张\n")
                return str(data_dir), images
        
        return None, []
    
    def _prepare_casia(self, max_samples):
        """Prepare CASIA-WebFace dataset"""
        sse_print("preparing_casia", {}, progress=22,
                 message="准备CASIA-WebFace数据集...",
                 log="[22%] 准备CASIA数据集\n")
        
        data_dir = self.data_path / 'data'
        if data_dir.exists():
            images = list(data_dir.rglob('*.jpg'))[:max_samples]
            if images:
                sse_print("casia_ready", {}, progress=24,
                         message=f"CASIA数据集就绪: {len(images)}张",
                         log=f"[24%] CASIA: {len(images)}张\n")
                return str(data_dir), images
        
        return None, []
    
    def _prepare_megaface(self, max_samples):
        """Prepare MegaFace dataset"""
        sse_print("preparing_megaface", {}, progress=22,
                 message="准备MegaFace数据集...",
                 log="[22%] 准备MegaFace数据集\n")
        
        data_dir = self.data_path / 'data'
        if data_dir.exists():
            images = list(data_dir.rglob('*.jpg'))[:max_samples]
            if images:
                sse_print("megaface_ready", {}, progress=24,
                         message=f"MegaFace数据集就绪: {len(images)}张",
                         log=f"[24%] MegaFace: {len(images)}张\n")
                return str(data_dir), images
        
        return None, []
    
    def _prepare_generic(self, max_samples):
        """Prepare generic image directory"""
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.pgm']:
            images.extend(list(self.data_path.rglob(ext)))
        images = [f for f in images if '__MACOSX' not in str(f)][:max_samples]
        
        if images:
            return str(self.data_path), images
        return None, []


def detect_and_prepare_dataset(data_path, max_samples=50):
    """
    Main function to detect and prepare any supported dataset
    Returns: (prepared_path, image_list, dataset_type)
    """
    detector = DatasetDetector(data_path)
    dataset_type = detector.detect_dataset_type()
    prepared_path, images = detector.prepare_dataset(max_samples)
    
    return prepared_path, images, dataset_type



