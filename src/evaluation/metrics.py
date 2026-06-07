from collections import defaultdict
from dataclasses import dataclass, field

UNKNOWN = "UNKNOWN"


@dataclass
class OWODResult:
    # Container for all OWOD evaluation numbers

    known_accuracy: float = 0.0
    unknown_recall: float = 0.0
    open_set_error_rate: float = 0.0
    a_ose: int = 0
    wilderness_impact: float = 0.0
    harmonic_mean: float = 0.0

    total_known_gt: int = 0
    total_unknown_gt: int = 0
    total_predictions: int = 0

    per_class: dict[str, dict] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "known_accuracy": round(self.known_accuracy, 4),
            "unknown_recall": round(self.unknown_recall, 4),
            "open_set_error_rate": round(self.open_set_error_rate, 4),
            "a_ose": self.a_ose,
            "wilderness_impact": round(self.wilderness_impact, 4),
            "harmonic_mean": round(self.harmonic_mean, 4),
            "total_known_gt": self.total_known_gt,
            "total_unknown_gt": self.total_unknown_gt,
            "total_predictions": self.total_predictions,
            "per_class": self.per_class,
            "confusion": self.confusion,
        }


class OWODMetrics:
    # Accumulates (pred, gt) label pairs and computes OWOD metrics

    def __init__(self, known_classes: list[str]):
        self.known_classes = list(known_classes)
        self._known_set = set(known_classes)

        self._pairs: list[tuple[str, str]] = []

    def add(self, pred_label: str, gt_label: str):
        self._pairs.append((pred_label, gt_label))

    def compute(self) -> OWODResult:
        all_labels = list(self.known_classes) + [UNKNOWN]

        confusion: dict[str, dict[str, int]] = {
            gt: defaultdict(int) for gt in all_labels
        }

        known_correct = 0
        known_total = 0
        unknown_correct = 0
        unknown_total = 0
        a_ose = 0

        per_cls_correct: dict[str, int] = defaultdict(int)
        per_cls_total: dict[str, int] = defaultdict(int)

        # FP tracking for Wilderness Impact
        fp_from_known = 0
        fp_from_unknown = 0
        tp_known = 0

        for pred, gt in self._pairs:
            gt_bucket = gt if gt in self._known_set else UNKNOWN
            confusion[gt_bucket][pred] += 1

            if gt_bucket != UNKNOWN:
                # GT is a known class
                known_total += 1
                per_cls_total[gt] += 1
                if pred == gt:
                    known_correct += 1
                    tp_known += 1
                    per_cls_correct[gt] += 1
                elif pred != UNKNOWN:
                    fp_from_known += 1
            else:
                # GT is unknown
                unknown_total += 1
                if pred == UNKNOWN:
                    unknown_correct += 1
                else:
                    a_ose += 1
                    fp_from_unknown += 1

        known_acc = known_correct / max(1, known_total)
        unk_recall = unknown_correct / max(1, unknown_total)
        ose_rate = a_ose / max(1, unknown_total)

        # Wilderness Impact: how much do unknowns degrade known-class precision
        precision_closed = tp_known / max(1, tp_known + fp_from_known)
        precision_open = tp_known / max(1, tp_known + fp_from_known + fp_from_unknown)
        wi = (precision_closed / max(1e-9, precision_open)) - 1.0

        # Harmonic mean of known accuracy and unknown recall
        hm = 2.0 * known_acc * unk_recall / max(1e-9, known_acc + unk_recall)

        # Per-class stats
        per_class = {}
        for cls in self.known_classes:
            t = per_cls_total[cls]
            c = per_cls_correct[cls]
            per_class[cls] = {
                "correct": c,
                "total": t,
                "accuracy": round(c / max(1, t), 4),
            }

        return OWODResult(
            known_accuracy=known_acc,
            unknown_recall=unk_recall,
            open_set_error_rate=ose_rate,
            a_ose=a_ose,
            wilderness_impact=wi,
            harmonic_mean=hm,
            total_known_gt=known_total,
            total_unknown_gt=unknown_total,
            total_predictions=len(self._pairs),
            per_class=per_class,
            confusion={k: dict(v) for k, v in confusion.items()},
        )

    def print_report(self, result=None):
        if result is None:
            result = self.compute()

        print(f"\n{'='*65}")
        print("  OWOD Evaluation Results")
        print(f"{'='*65}")
        print(f"  Known classes          : {len(self.known_classes)}")
        print(f"  Total predictions      : {result.total_predictions}")
        print(f"  Known GT objects       : {result.total_known_gt}")
        print(f"  Unknown GT objects     : {result.total_unknown_gt}")
        print(f"{'─'*65}")
        print(f"  Known Accuracy         : {result.known_accuracy:.1%}")
        print(f"  Unknown Recall         : {result.unknown_recall:.1%}")
        print(f"  Harmonic Mean (H)      : {result.harmonic_mean:.1%}")
        print(f"{'─'*65}")
        print(f"  A-OSE (abs. count)     : {result.a_ose}")
        print(f"  Open-Set Error Rate    : {result.open_set_error_rate:.1%}")
        print(f"  Wilderness Impact (WI) : {result.wilderness_impact:.4f}")
        print(f"{'='*65}")

        print(f"\n{'Class':<20} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print("-" * 48)
        for cls in self.known_classes:
            info = result.per_class.get(cls, {})
            t = info.get("total", 0)
            c = info.get("correct", 0)
            acc = f"{c / t:.1%}" if t > 0 else "n/a"
            print(f"{cls:<20} {c:>8} {t:>8} {acc:>8}")
