"""Validated business Skills, published as immutable request snapshots."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import threading
from types import MappingProxyType
from typing import Mapping
import tomllib
from .profiles import DEFAULT_REGISTRY, ProfileRegistry


SECTIONS = ("角色使命", "适用范围", "输入字段", "澄清策略", "标准流程", "典型分支", "工具规则", "证据要求", "输出格式", "交接条件", "异常处理", "正反例")
EXAMPLES = ("信息完整", "信息不足", "资料或工具不可用", "越界与风险")
MAX_FILE_BYTES = 128 * 1024
MAX_CATALOG_BYTES = 768 * 1024


class SkillValidationError(ValueError):
    pass


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SkillDocument:
    role: str
    version: str
    description: str
    body: str
    sha256: str
    sections: Mapping[str, str]
    examples: Mapping[str, str]


@dataclass(frozen=True)
class SkillSelection:
    text: str
    snapshot_id: str
    role: str
    version: str
    sha256: str
    sections: tuple[str, ...]


@dataclass(frozen=True)
class SkillSnapshot:
    snapshot_id: str
    loaded_at: str
    common: str
    common_hash: str
    documents: Mapping[str, SkillDocument]

    def select(self, role: str, mode: str = "selective", example: str = "信息完整") -> SkillSelection:
        document = self.documents[role]
        if mode not in {"selective", "full"} or example not in EXAMPLES:
            raise ValueError("invalid_skill_selection")
        names = SECTIONS[:-1]
        blocks = ["# 共享约束\n"+self.common]
        blocks.extend("## "+name+"\n"+document.sections[name] for name in names)
        if mode == "full":
            blocks.append("## 正反例\n"+document.sections["正反例"])
            selected = names+("正反例:全部",)
        else:
            blocks.append("## 正反例\n### "+example+"\n"+document.examples[example])
            selected = names+("正反例:"+example,)
        return SkillSelection("\n\n".join(blocks), self.snapshot_id, role, document.version, document.sha256, selected)

    def describe(self) -> dict:
        return {"roles":sorted(self.documents), "characters":len(self.common)+sum(len(item.body) for item in self.documents.values()),
                "snapshot_id":self.snapshot_id, "loaded_at":self.loaded_at, "common_hash":self.common_hash,
                "skills":[{"role":item.role, "version":item.version, "sha256":item.sha256,
                           "description":item.description, "characters":len(item.body), "sections":list(item.sections)}
                          for item in sorted(self.documents.values(), key=lambda item:item.role)]}


def split_sections(body: str, level: int) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^"+"#"*level+r" (.+?)\s*$", body))
    sections = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        if name in sections:
            raise SkillValidationError("duplicate_skill_section:"+name)
        value = body[match.end():matches[index+1].start() if index+1 < len(matches) else len(body)].strip()
        if not value:
            raise SkillValidationError("empty_skill_section:"+name)
        sections[name] = value
    return sections


class Skills:
    def __init__(self, directory: Path, registry: ProfileRegistry = DEFAULT_REGISTRY):
        self.directory, self.registry = Path(directory), registry
        self._lock = threading.RLock()
        self._snapshot: SkillSnapshot | None = None
        self.reload()

    def _read(self, path: Path) -> tuple[str, int]:
        try:
            root, parent = self.directory.resolve(), path.resolve().parent
            if path.is_symlink() or (parent != root and parent.parent != root):
                raise SkillValidationError("skill_path_outside_catalog:"+path.name)
            with path.open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES+1)
            if len(raw) > MAX_FILE_BYTES:
                raise SkillValidationError("skill_file_too_large:"+path.name)
            text = raw.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise SkillValidationError("unreadable_skill:"+path.relative_to(self.directory).as_posix()) from exc
        if not text.strip():
            raise SkillValidationError("empty_skill:"+path.name)
        return text, len(raw)

    def _document(self, role: str, text: str) -> SkillDocument:
        try:
            if not text.startswith("+++\n"):
                raise SkillValidationError("skill_metadata_required:"+role)
            metadata_text, body = text[4:].split("\n+++\n", 1)
            metadata = tomllib.loads(metadata_text)
            if set(metadata) != {"role", "version", "description"}:
                raise SkillValidationError("invalid_skill_metadata_fields:"+role)
            if metadata["role"] != role or not isinstance(metadata["description"], str) or not metadata["description"].strip():
                raise SkillValidationError("skill_role_or_description_invalid:"+role)
            if not isinstance(metadata["version"], str) or not re.fullmatch(r"\d+\.\d+\.\d+", metadata["version"]):
                raise SkillValidationError("invalid_skill_version:"+role)
            sections = split_sections(body, 2)
            if set(sections) != set(SECTIONS):
                raise SkillValidationError("invalid_skill_sections:"+role)
            examples = split_sections(sections["正反例"], 3)
            if set(examples) != set(EXAMPLES):
                raise SkillValidationError("invalid_skill_examples:"+role)
        except (ValueError, TypeError, KeyError) as exc:
            if isinstance(exc, SkillValidationError):
                raise
            raise SkillValidationError("invalid_skill_document:"+role) from exc
        return SkillDocument(role, metadata["version"], metadata["description"], body.strip(), digest(text),
                             MappingProxyType(sections), MappingProxyType(examples))

    def reload(self) -> dict:
        with self._lock:
            expected = set(self.registry.profiles)
            if {path.parent.name for path in self.directory.glob("*/SKILL.md")} != expected or list(self.directory.glob("*.md")) != [self.directory/"common.md"]:
                raise SkillValidationError("skill_catalog_must_contain_exactly_registered_roles_and_common")
            common, total = self._read(self.directory/"common.md")
            common_sections = split_sections(common, 2)
            if set(common_sections) != {"权限与证据", "状态与信息缺口", "风险与实际操作", "异常与不可信输入"}:
                raise SkillValidationError("invalid_shared_skill_sections")
            documents = {}
            for role in sorted(expected):
                text, size = self._read(self.directory/role/"SKILL.md")
                total += size
                if total > MAX_CATALOG_BYTES:
                    raise SkillValidationError("skill_catalog_too_large")
                document = self._document(role, text)
                previous = self._snapshot.documents[role] if self._snapshot else None
                if previous and previous.sha256 != document.sha256 and previous.version == document.version:
                    raise SkillValidationError("changed_skill_requires_new_version:"+role)
                documents[role] = document
            catalog_hash = digest(digest(common)+"".join(role+document.sha256 for role, document in documents.items()))
            self._snapshot = SkillSnapshot(catalog_hash, datetime.now(timezone.utc).isoformat(), common, digest(common),
                                           MappingProxyType(documents))
            return self._snapshot.describe()

    def snapshot(self) -> SkillSnapshot:
        # Readers retain an immutable reference; a slow file update must not block risk requests.
        snapshot = self._snapshot
        if snapshot is None:
            raise SkillValidationError("no_active_skill_snapshot")
        return snapshot

