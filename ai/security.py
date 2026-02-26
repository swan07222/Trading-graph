"""Security module with prompt injection detection.

Fixes:
- Prompt injection attacks: Multi-layer detection and prevention
- Security vulnerabilities: Input sanitization and output filtering
- Unauthorized access: Command-level authorization
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class SecurityLevel(Enum):
    """Security levels for prompt processing."""
    NONE = auto()       # No security checks
    LOW = auto()        # Basic pattern matching
    MEDIUM = auto()     # Pattern + semantic analysis
    HIGH = auto()       # Full analysis with logging
    MAXIMUM = auto()    # Maximum security, block suspicious


class ThreatType(Enum):
    """Types of security threats."""
    PROMPT_INJECTION = auto()
    JAILBREAK_ATTEMPT = auto()
    DATA_EXFILTRATION = auto()
    AUTHORIZATION_BYPASS = auto()
    MALICIOUS_CODE = auto()
    SOCIAL_ENGINEERING = auto()
    UNKNOWN = auto()


@dataclass
class SecurityThreat:
    """A detected security threat."""
    threat_id: str
    threat_type: ThreatType
    severity: int  # 1-10
    description: str
    original_input: str
    matched_patterns: list[str]
    confidence: float  # 0-1
    timestamp: datetime = field(default_factory=datetime.now)
    blocked: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "threat_id": self.threat_id,
            "threat_type": self.threat_type.name,
            "severity": self.severity,
            "description": self.description,
            "matched_patterns": self.matched_patterns,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "blocked": self.blocked,
        }


@dataclass
class SecurityReport:
    """Security analysis report for input/output."""
    input_text: str
    security_level: SecurityLevel
    is_safe: bool
    threats: list[SecurityThreat] = field(default_factory=list)
    sanitized_text: str = ""
    recommendations: list[str] = field(default_factory=list)
    analysis_time_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "input_text_preview": self.input_text[:200],
            "security_level": self.security_level.name,
            "is_safe": self.is_safe,
            "threats": [t.to_dict() for t in self.threats],
            "sanitized_text": self.sanitized_text,
            "recommendations": self.recommendations,
            "analysis_time_ms": self.analysis_time_ms,
        }


class PromptGuard:
    """Multi-layer prompt injection detection and prevention.
    
    Detection Layers:
    1. Pattern Matching - Known attack patterns
    2. Semantic Analysis - Intent classification
    3. Context Validation - Command context verification
    4. Output Filtering - Response sanitization
    """
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        # Direct injection attempts
        r"ignore\s+(?:previous|all|above|prior)\s+(?:instructions|rules|constraints)",
        r"(?:forget|disregard|ignore)\s+(?:your|the)\s+(?:instructions|rules|guidelines|constraints)",
        r"(?:you\s+are\s+)?(?:now|from\s+now|henceforth)\s+(?:free|unrestricted|unlimited)",
        r"(?:act\s+as|pretend\s+to\s+be|roleplay)\s+(?:a\s+)?(?:hacker|developer|admin|system)",
        r"(?:bypass|circumvent|skip|override)\s+(?:your|the)\s+(?:safety|security|ethical|content)\s+(?:filters|rules|policies)",
        
        # DAN-style jailbreaks
        r"(?:do\s+anything\s+now|dan|developer\s+mode|god\s+mode)",
        r"(?:you\s+are\s+dan|act\s+like\s+dan)",
        
        # Authority impersonation
        r"(?:i\s+am\s+your\s+)?(?:creator|developer|admin|owner|manager)",
        r"(?:this\s+is\s+an\s+)?(?:emergency|critical|urgent|official)\s+(?:request|command|order)",
        r"(?:authorized\s+by|approved\s+by|permission\s+from)\s+(?:admin|developer|ceo)",
        
        # Data exfiltration
        r"(?:reveal|disclose|output|print|show)\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules|configuration)",
        r"(?:what\s+is\s+your|tell\s+me\s+your|show\s+me\s+your)\s+(?:system\s+)?(?:prompt|instructions)",
        r"(?:export|leak|share)\s+(?:your\s+)?(?:training\s+)?(?:data|knowledge)",
        
        # Code execution
        r"(?:execute|run|eval|interpret)\s+(?:code|script|command|python|sql)",
        r"(?:system\s+)?(?:command|shell|exec)\s*[:\(]",
        
        # Hypnosis/manipulation
        r"(?:you\s+are\s+)?(?:hypnotized|tricked|convinced|persuaded)",
        r"(?:from\s+now\s+on|imagine\s+that|suppose\s+that|pretend\s+that)",
    ]
    
    # Safe patterns (whitelist for common phrases)
    SAFE_PATTERNS = [
        r"please\s+help\s+me",
        r"can\s+you\s+(?:tell|explain|show)",
        r"what\s+is\s+(?:the|a)",
        r"how\s+to\s+\w+",
    ]
    
    # Sensitive topics that require extra scrutiny
    SENSITIVE_TOPICS = [
        "password", "credential", "api\s*key", "token", "secret",
        "private\s*key", "encryption\s*key", "database\s*password",
        "admin\s*panel", "backdoor", "exploit", "vulnerability",
    ]
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        block_threshold: float = 0.7,
        log_dir: Path | None = None,
    ) -> None:
        self.security_level = security_level
        self.block_threshold = block_threshold
        self.log_dir = log_dir or (CONFIG.logs_dir / "security")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Compile patterns
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        self._safe_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SAFE_PATTERNS
        ]
        self._sensitive_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SENSITIVE_TOPICS
        ]
        
        # Threat history for rate limiting
        self._threat_history: list[tuple[datetime, str]] = []
        self._blocked_hashes: set[str] = set()
        
        log.info(f"PromptGuard initialized: {security_level.name}")
    
    def analyze(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> SecurityReport:
        """Analyze text for security threats.
        
        Args:
            text: Input text to analyze
            context: Optional context (user role, command type, etc.)
            
        Returns:
            SecurityReport with analysis results
        """
        import time
        start = time.time()
        
        threats = []
        recommendations = []
        
        # Skip analysis if security level is NONE
        if self.security_level == SecurityLevel.NONE:
            return SecurityReport(
                input_text=text,
                security_level=self.security_level,
                is_safe=True,
                sanitized_text=text,
                analysis_time_ms=(time.time() - start) * 1000,
            )
        
        # Layer 1: Pattern matching
        pattern_threats = self._check_patterns(text)
        threats.extend(pattern_threats)
        
        # Layer 2: Semantic analysis (if medium+)
        if self.security_level >= SecurityLevel.MEDIUM:
            semantic_threats = self._check_semantic(text, context)
            threats.extend(semantic_threats)
        
        # Layer 3: Context validation (if high+)
        if self.security_level >= SecurityLevel.HIGH:
            context_threats = self._check_context(text, context or {})
            threats.extend(context_threats)
        
        # Determine if safe
        max_severity = max((t.severity for t in threats), default=0)
        max_confidence = max((t.confidence for t in threats), default=0)
        
        is_safe = (
            max_severity < 5 and
            max_confidence < self.block_threshold
        )
        
        # Generate sanitized text
        sanitized_text = self._sanitize_text(text) if not is_safe else text
        
        # Generate recommendations
        if threats:
            recommendations = self._generate_recommendations(threats)
        
        # Log threat if detected
        if threats and self.security_level >= SecurityLevel.HIGH:
            self._log_threats(threats)
        
        return SecurityReport(
            input_text=text,
            security_level=self.security_level,
            is_safe=is_safe,
            threats=threats,
            sanitized_text=sanitized_text,
            recommendations=recommendations,
            analysis_time_ms=(time.time() - start) * 1000,
        )
    
    def _check_patterns(self, text: str) -> list[SecurityThreat]:
        """Check for known attack patterns."""
        threats = []
        import uuid
        
        text_lower = text.lower()
        
        # Check if text matches safe patterns (early exit)
        for pattern in self._safe_patterns:
            if pattern.search(text):
                # Still check for sensitive topics
                pass
        
        # Check injection patterns
        matched = []
        for i, pattern in enumerate(self._injection_patterns):
            if pattern.search(text):
                matched.append(pattern.pattern[:50])
        
        if matched:
            # Calculate confidence based on number of matches
            confidence = min(1.0, len(matched) * 0.3)
            
            # Check for sensitive topics
            sensitive_matched = []
            for pattern in self._sensitive_patterns:
                if pattern.search(text):
                    sensitive_matched.append(pattern.pattern)
            
            if sensitive_matched:
                confidence = min(1.0, confidence + 0.2)
            
            threats.append(SecurityThreat(
                threat_id=f"threat_{uuid.uuid4().hex[:8]}",
                threat_type=ThreatType.PROMPT_INJECTION,
                severity=6 if not sensitive_matched else 8,
                description=f"Detected {len(matched)} injection pattern(s)",
                original_input=text[:500],
                matched_patterns=matched,
                confidence=confidence,
            ))
        
        return threats
    
    def _check_semantic(
        self,
        text: str,
        context: dict[str, Any] | None,
    ) -> list[SecurityThreat]:
        """Semantic analysis for jailbreak attempts."""
        threats = []
        import uuid
        
        text_lower = text.lower()
        
        # Check for jailbreak keywords
        jailbreak_keywords = [
            "dan", "developer mode", "god mode", "unrestricted",
            "without restrictions", "ignore rules", "break rules",
        ]
        
        matched_keywords = [kw for kw in jailbreak_keywords if kw in text_lower]
        
        if matched_keywords:
            threats.append(SecurityThreat(
                threat_id=f"threat_{uuid.uuid4().hex[:8]}",
                threat_type=ThreatType.JAILBREAK_ATTEMPT,
                severity=7,
                description=f"Jailbreak keywords detected: {', '.join(matched_keywords)}",
                original_input=text[:500],
                matched_patterns=matched_keywords,
                confidence=0.8,
            ))
        
        # Check for instruction extraction attempts
        extraction_phrases = [
            "what are your instructions",
            "what is your prompt",
            "what were you told",
            "repeat the above",
            "output your system message",
        ]
        
        for phrase in extraction_phrases:
            if phrase in text_lower:
                threats.append(SecurityThreat(
                    threat_id=f"threat_{uuid.uuid4().hex[:8]}",
                    threat_type=ThreatType.DATA_EXFILTRATION,
                    severity=6,
                    description=f"Instruction extraction attempt: '{phrase}'",
                    original_input=text[:500],
                    matched_patterns=[phrase],
                    confidence=0.75,
                ))
                break
        
        return threats
    
    def _check_context(
        self,
        text: str,
        context: dict[str, Any],
    ) -> list[SecurityThreat]:
        """Context-aware validation."""
        threats = []
        import uuid
        
        # Check for authorization bypass attempts
        if context.get("command_type") in ["BUY", "SELL", "TRANSFER"]:
            auth_phrases = [
                "ignore confirmation",
                "skip verification",
                "execute immediately",
                "don't ask",
                "just do it",
            ]
            
            text_lower = text.lower()
            for phrase in auth_phrases:
                if phrase in text_lower:
                    threats.append(SecurityThreat(
                        threat_id=f"threat_{uuid.uuid4().hex[:8]}",
                        threat_type=ThreatType.AUTHORIZATION_BYPASS,
                        severity=9,
                        description=f"Authorization bypass attempt in trading command",
                        original_input=text[:500],
                        matched_patterns=[phrase],
                        confidence=0.9,
                    ))
                    break
        
        # Check for rate limiting
        user_id = context.get("user_id", "unknown")
        recent_threats = sum(
            1 for ts, uid in self._threat_history
            if uid == user_id and
            (datetime.now() - ts).total_seconds() < 60
        )
        
        if recent_threats >= 5:
            threats.append(SecurityThreat(
                threat_id=f"threat_{uuid.uuid4().hex[:8]}",
                threat_type=ThreatType.UNKNOWN,
                severity=5,
                description=f"Rate limit exceeded: {recent_threats} threats in 60s",
                original_input=text[:500],
                matched_patterns=["rate_limit"],
                confidence=1.0,
            ))
        
        return threats
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing potentially dangerous content."""
        # Remove common injection prefixes
        sanitized = text
        
        # Remove "ignore previous instructions" variants
        sanitized = re.sub(
            r"(?i)(ignore|forget|disregard)\s+(previous|all|above|prior|your|the)\s+(instructions|rules|guidelines|constraints|prompt)",
            "[REMOVED]",
            sanitized,
        )
        
        # Remove roleplay attempts
        sanitized = re.sub(
            r"(?i)(act\s+as|pretend\s+to\s+be|roleplay|you\s+are\s+now)\s+(a\s+)?(developer|admin|hacker|system)",
            "[REMOVED]",
            sanitized,
        )
        
        return sanitized
    
    def _generate_recommendations(
        self,
        threats: list[SecurityThreat],
    ) -> list[str]:
        """Generate security recommendations."""
        recommendations = []
        
        threat_types = set(t.threat_type for t in threats)
        
        if ThreatType.PROMPT_INJECTION in threat_types:
            recommendations.append(
                "Input contains prompt injection patterns. Consider blocking or sanitizing."
            )
        
        if ThreatType.JAILBREAK_ATTEMPT in threat_types:
            recommendations.append(
                "Jailbreak attempt detected. Block request and consider rate limiting user."
            )
        
        if ThreatType.DATA_EXFILTRATION in threat_types:
            recommendations.append(
                "Data exfiltration attempt. Never reveal system prompts or internal configurations."
            )
        
        if ThreatType.AUTHORIZATION_BYPASS in threat_types:
            recommendations.append(
                "Authorization bypass attempt in sensitive operation. Require additional confirmation."
            )
        
        max_severity = max((t.severity for t in threats), default=0)
        if max_severity >= 8:
            recommendations.append(
                "High severity threat detected. Consider logging and alerting security team."
            )
        
        return recommendations
    
    def _log_threats(self, threats: list[SecurityThreat]) -> None:
        """Log detected threats."""
        log_file = self.log_dir / f"threats_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                for threat in threats:
                    line = json.dumps(threat.to_dict(), ensure_ascii=False)
                    f.write(line + "\n")
                    
                    # Update history for rate limiting
                    self._threat_history.append((datetime.now(), "unknown"))
                    
        except Exception as e:
            log.error(f"Failed to log threat: {e}")
    
    def is_blocked(self, text: str) -> bool:
        """Quick check if text should be blocked."""
        # Check hash cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._blocked_hashes:
            return True
        
        # Quick pattern check
        for pattern in self._injection_patterns:
            if pattern.search(text):
                self._blocked_hashes.add(text_hash)
                return True
        
        return False
    
    def get_stats(self) -> dict[str, Any]:
        """Get security statistics."""
        return {
            "security_level": self.security_level.name,
            "block_threshold": self.block_threshold,
            "pattern_count": len(self._injection_patterns),
            "blocked_cache_size": len(self._blocked_hashes),
            "recent_threats": len(self._threat_history),
        }


# JSON helper for logging
import json


# Singleton instance
_guard_instance: PromptGuard | None = None


def get_prompt_guard(
    security_level: SecurityLevel | None = None,
) -> PromptGuard:
    """Get or create the singleton PromptGuard instance."""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = PromptGuard(security_level or SecurityLevel.HIGH)
    return _guard_instance


def check_prompt_security(
    text: str,
    context: dict[str, Any] | None = None,
) -> SecurityReport:
    """Convenience function to check prompt security."""
    return get_prompt_guard().analyze(text, context)
