import codecs


class TextLoader:
    SUPPORTED_EXTENSIONS: set[str] = {".txt", ".epub", ".xml"}

    def load(self, path: str) -> str:
        ext = path.lower()
        if not any(path.endswith(e) for e in self.SUPPORTED_EXTENSIONS):
            raise ValueError(f"Unsupported extension: {path}. Must be one of {self.SUPPORTED_EXTENSIONS}")
        if path.endswith(".txt"):
            return self._load_txt(path)
        elif path.endswith(".xml"):
            return self._load_xml(path)
        elif path.endswith(".epub"):
            return self._load_epub(path)
        raise ValueError(f"Cannot load: {path}")

    def _load_txt(self, path: str) -> str:
        encoding = self.detect_encoding(path)
        with open(path, "r", encoding=encoding) as f:
            return f.read()

    def _load_xml(self, path: str) -> str:
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        texts = []
        for elem in root.iter("{http://www.tei-c.org/ns/1.0}text"):
            texts.append("".join(elem.itertext()))
        if not texts:
            texts.append("".join(root.itertext()))
        return "\n\n".join(texts)

    def _load_epub(self, path: str) -> str:
        try:
            import ebooklib
            from ebooklib import epub
            book = epub.read_epub(path)
            texts = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    texts.append(item.get_content().decode("utf-8"))
            return "\n\n".join(texts)
        except ImportError:
            raise ImportError("ebooklib is required for EPUB files. Install with: pip install ebooklib")

    def detect_encoding(self, path: str) -> str:
        try:
            import chardet
            with open(path, "rb") as f:
                raw = f.read(10000)
                result = chardet.detect(raw)
                return result["encoding"] or "utf-8"
        except ImportError:
            with open(path, "rb") as f:
                raw = f.read(4)
            if raw.startswith(b"\xff\xfe"):
                return "utf-16-le"
            elif raw.startswith(b"\xfe\xff"):
                return "utf-16-be"
            elif raw.startswith(b"\xef\xbb\xbf"):
                return "utf-8-sig"
            return "utf-8"
