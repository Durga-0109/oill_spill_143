import datetime
import json
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

class AuditService:
    """
    Forensic Audit Trail Logger.
    Logs system operations with timestamp, operator ID, action type, and analysis ID.
    """

    def __init__(self):
        self.in_memory_logs = []

    def log(
        self,
        action: str,
        analysis_id: Optional[str] = None,
        operator: str = "OPERATOR-01",
        details: Optional[Dict[str, Any]] = None,
        db_session: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Creates an audit trail entry.
        """
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "operator": operator,
            "action": action,
            "analysis_id": analysis_id or "SYSTEM",
            "details": details or {}
        }

        self.in_memory_logs.append(entry)
        if len(self.in_memory_logs) > 500:
            self.in_memory_logs.pop(0)

        # Write to DB if session provided
        if db_session:
            try:
                from database import AuditLog
                db_record = AuditLog(
                    operator=operator,
                    action=action,
                    analysis_id=analysis_id,
                    details_json=json.dumps(details or {}),
                    created_at=datetime.datetime.utcnow()
                )
                db_session.add(db_record)
                db_session.commit()
            except Exception:
                pass

        return entry

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(reversed(self.in_memory_logs[-limit:]))

audit_service = AuditService()
