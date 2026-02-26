# scripts/todo_resolver.py
"""
TODO/FIXME Resolution Script

FIXES:
- Track and resolve TODO/FIXME/XXX/HACK comments
- Generate technical debt report
- Prioritize and assign resolution tasks
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TechnicalDebt:
    """Technical debt item."""
    file_path: str
    line_number: int
    debt_type: str  # TODO, FIXME, XXX, HACK, BUG
    message: str
    priority: str = "medium"  # low, medium, high, critical
    created_date: Optional[str] = None
    assignee: Optional[str] = None
    status: str = "open"  # open, in_progress, resolved
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "debt_type": self.debt_type,
            "message": self.message,
            "priority": self.priority,
            "created_date": self.created_date,
            "assignee": self.assignee,
            "status": self.status,
        }


class TechnicalDebtTracker:
    """
    Track and manage technical debt from code comments.
    
    FIXES:
    1. Centralized TODO/FIXME tracking
    2. Priority assignment based on context
    3. Resolution workflow
    4. Progress reporting
    """
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.debts: list[TechnicalDebt] = []
        self.pattern = re.compile(
            r"(TODO|FIXME|XXX|HACK|BUG)[:\s]*(.+?)(?=\n\s*(?:#|//|/\\*|$))",
            re.IGNORECASE | re.DOTALL,
        )
    
    def scan_directory(
        self,
        patterns: Optional[list[str]] = None,
        exclude_dirs: Optional[list[str]] = None,
    ) -> list[TechnicalDebt]:
        """
        Scan directory for technical debt comments.
        
        FIX: Automated debt discovery
        """
        if patterns is None:
            patterns = ["*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.h"]
        
        if exclude_dirs is None:
            exclude_dirs = [
                "venv", ".venv", "__pycache__", ".git",
                "node_modules", "build", "dist",
                "models_saved", "cache", "logs",
            ]
        
        self.debts = []
        
        for pattern in patterns:
            for file_path in self.root_path.rglob(pattern):
                # Skip excluded directories
                if any(excluded in str(file_path) for excluded in exclude_dirs):
                    continue
                
                self._scan_file(file_path)
        
        log.info(f"Found {len(self.debts)} technical debt items")
        return self.debts
    
    def _scan_file(self, file_path: Path) -> None:
        """Scan single file for debt comments."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            lines = content.split("\n")
            
            for match in self.pattern.finditer(content):
                # Calculate line number
                line_number = content[:match.start()].count("\n") + 1
                
                debt_type = match.group(1).upper()
                message = match.group(2).strip()
                
                # Determine priority based on type and content
                priority = self._determine_priority(debt_type, message)
                
                debt = TechnicalDebt(
                    file_path=str(file_path),
                    line_number=line_number,
                    debt_type=debt_type,
                    message=message,
                    priority=priority,
                )
                self.debts.append(debt)
        
        except Exception as e:
            log.warning(f"Failed to scan {file_path}: {e}")
    
    def _determine_priority(
        self,
        debt_type: str,
        message: str,
    ) -> str:
        """Determine priority based on debt type and message."""
        message_lower = message.lower()
        
        # Critical: Security, data loss, crashes
        critical_keywords = [
            "security", "crash", "data loss", "corrupt",
            "leak", "vulnerability", "critical", "urgent",
        ]
        if any(kw in message_lower for kw in critical_keywords):
            return "critical"
        
        # High: BUG, FIXME with error-related context
        if debt_type in ["BUG", "FIXME"]:
            high_keywords = [
                "error", "fail", "exception", "broken",
                "not working", "incorrect", "wrong",
            ]
            if any(kw in message_lower for kw in high_keywords):
                return "high"
        
        # Medium: TODO, HACK
        if debt_type in ["TODO", "HACK"]:
            return "medium"
        
        # Low: XXX, general improvements
        return "low"
    
    def generate_report(
        self,
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate technical debt report.
        
        FIX: Visibility into technical debt
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_items": len(self.debts),
                "by_type": {},
                "by_priority": {},
                "by_status": {},
                "by_file": {},
            },
            "items": [debt.to_dict() for debt in self.debts],
            "recommendations": [],
        }
        
        # Count by type
        for debt in self.debts:
            report["summary"]["by_type"][debt.debt_type] = (
                report["summary"]["by_type"].get(debt.debt_type, 0) + 1
            )
            report["summary"]["by_priority"][debt.priority] = (
                report["summary"]["by_priority"].get(debt.priority, 0) + 1
            )
            report["summary"]["by_status"][debt.status] = (
                report["summary"]["by_status"].get(debt.status, 0) + 1
            )
            
            # Group by file
            file_key = debt.file_path
            if file_key not in report["summary"]["by_file"]:
                report["summary"]["by_file"][file_key] = []
            report["summary"]["by_file"][file_key].append(debt.to_dict())
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()
        
        # Save report
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            log.info(f"Report saved to {output_path}")
        
        return report
    
    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Count by priority
        critical_count = sum(1 for d in self.debts if d.priority == "critical")
        high_count = sum(1 for d in self.debts if d.priority == "high")
        
        if critical_count > 0:
            recommendations.append(
                f"IMMEDIATE: Address {critical_count} critical issues first"
            )
        
        if high_count > 0:
            recommendations.append(
                f"SHORT-TERM: Resolve {high_count} high-priority issues within 2 weeks"
            )
        
        # Group by file
        files_with_many_issues = {}
        for debt in self.debts:
            files_with_many_issues[debt.file_path] = (
                files_with_many_issues.get(debt.file_path, 0) + 1
            )
        
        for file_path, count in files_with_many_issues.items():
            if count >= 5:
                recommendations.append(
                    f"REFACTOR: {file_path} has {count} issues - consider refactoring"
                )
        
        # Type-specific recommendations
        hack_count = sum(1 for d in self.debts if d.debt_type == "HACK")
        if hack_count > 3:
            recommendations.append(
                f"CLEANUP: Replace {hack_count} HACK workarounds with proper solutions"
            )
        
        bug_count = sum(1 for d in self.debts if d.debt_type == "BUG")
        if bug_count > 0:
            recommendations.append(
                f"QUALITY: Fix {bug_count} known bugs before next release"
            )
        
        return recommendations
    
    def resolve_debt(
        self,
        file_path: str,
        line_number: int,
        resolution_note: str = "",
    ) -> bool:
        """
        Mark technical debt as resolved.
        
        FIX: Resolution workflow
        """
        for debt in self.debts:
            if debt.file_path == file_path and debt.line_number == line_number:
                debt.status = "resolved"
                debt.resolution_note = resolution_note
                debt.resolved_date = datetime.now().isoformat()
                log.info(f"Resolved debt: {debt.debt_type} at {file_path}:{line_number}")
                return True
        
        log.warning(f"Debt not found: {file_path}:{line_number}")
        return False
    
    def get_priority_summary(self) -> dict[str, int]:
        """Get summary by priority."""
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for debt in self.debts:
            summary[debt.priority] = summary.get(debt.priority, 0) + 1
        return summary


def main() -> int:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Technical Debt Tracker"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan for technical debt",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="technical_debt_report.json",
        help="Output path for report",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to scan",
    )
    parser.add_argument(
        "--min-priority",
        type=str,
        choices=["low", "medium", "high", "critical"],
        default="low",
        help="Minimum priority to include",
    )
    
    args = parser.parse_args()
    
    tracker = TechnicalDebtTracker(root_path=args.root)
    
    if args.scan:
        print("Scanning for technical debt...")
        debts = tracker.scan_directory()
        
        print(f"\nFound {len(debts)} technical debt items")
        
        # Generate report
        report = tracker.generate_report(output_path=args.report)
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Total: {report['summary']['total_items']}")
        print("\nBy Type:")
        for debt_type, count in report["summary"]["by_type"].items():
            print(f"  {debt_type}: {count}")
        print("\nBy Priority:")
        for priority, count in report["summary"]["by_priority"].items():
            print(f"  {priority}: {count}")
        
        print("\n=== Recommendations ===")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
        
        print(f"\nFull report saved to: {args.report}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
