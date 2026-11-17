"""
Executive AI Chief of Staff - Main Training Script
Orchestrates: data generation → preprocessing → annotation → DB storage →
feature extraction → model training → evaluation → intelligence processing.
"""
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config.settings import PROCESSED_DATA_DIR, MODELS_DIR, RAW_DATA_DIR, DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("TrainPipeline")


def main():
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Executive AI Chief of Staff - Training Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Initialize Database ──
    logger.info("\n📦 Step 1: Initializing database...")
    from src.database.db_manager import DatabaseManager
    db = DatabaseManager()
    db.init_db()
    stats = db.get_stats()
    logger.info(f"Database stats: {json.dumps({k: v for k, v in stats.items() if k not in ('priority_breakdown','commitment_breakdown','risk_breakdown','annotation_breakdown')}, default=str)}")

    # ── Step 2: Generate Dataset ──
    logger.info("\n📊 Step 2: Generating synthetic executive emails...")
    from src.data.dataset_loader import generate_synthetic_executive_emails, create_train_val_test_split, get_dataset_stats

    existing_count = stats.get('emails', 0) or stats.get('email_count', 0)
    if existing_count >= 100:
        logger.info(f"Using existing {existing_count} emails from database")
        raw_emails = db.get_emails({})
        emails = []
        for e in raw_emails:
            emails.append({
                'id': e.id,
                'db_id': e.id,
                'email_id': e.id,
                'message_id': e.message_id,
                'subject': e.subject,
                'sender': e.sender,
                'recipients': json.loads(e.recipients) if e.recipients else [],
                'cc': json.loads(e.cc) if e.cc else [],
                'date': e.date.isoformat() if e.date else None,
                'body': e.body,
                'in_reply_to': e.in_reply_to,
                'thread_id': e.thread_id,
                'is_reply': e.is_reply,
            })
    else:
        emails = generate_synthetic_executive_emails(n=3000)
        logger.info(f"Generated {len(emails)} synthetic emails")
        ds_stats = get_dataset_stats(emails)
        logger.info(f"Dataset: {json.dumps(ds_stats, default=str)}")
        # Save raw
        raw_path = RAW_DATA_DIR / "synthetic_emails.json"
        with open(raw_path, 'w') as f:
            json.dump(emails, f, indent=2, default=str)
        logger.info(f"Raw emails saved to {raw_path}")

    # ── Step 3: Preprocess ──
    logger.info("\n🔧 Step 3: Preprocessing emails...")
    from src.data.preprocessor import preprocess_email, extract_email_metadata
    for email in emails:
        try:
            email['cleaned_body'] = preprocess_email(email.get('body', ''))
            email['metadata'] = extract_email_metadata(email)
        except Exception as e:
            email['cleaned_body'] = email.get('body', '')
            email['metadata'] = {}
    logger.info(f"Preprocessed {len(emails)} emails")

    # ── Step 4: Annotate ──
    logger.info("\n🏷️ Step 4: Running annotation pipeline...")
    from src.data.annotation_pipeline import run_annotation_pipeline, save_annotations
    annotated_emails = run_annotation_pipeline(emails)
    ann_path = PROCESSED_DATA_DIR / "annotations.json"
    save_annotations(annotated_emails, str(ann_path))
    logger.info(f"Annotations saved to {ann_path}")

    # ── Step 5: Store in Database ──
    logger.info("\n💾 Step 5: Storing emails in database...")
    if existing_count < 100:
        count = db.bulk_add_emails(annotated_emails)
        logger.info(f"Stored {count} emails in database")
    else:
        logger.info("Emails already in database.")

    # Map all emails to their DB id by message_id
    all_db_emails = db.get_emails(limit=10000)
    msg_to_id = {e.message_id: e.id for e in all_db_emails if e.message_id}
    for email in annotated_emails:
        msg_id = email.get('message_id')
        if msg_id and msg_id in msg_to_id:
            email['id'] = msg_to_id[msg_id]
            email['db_id'] = msg_to_id[msg_id]
            email['email_id'] = msg_to_id[msg_id]
    logger.info(f"Mapped {len(msg_to_id)} database IDs to emails")

    # ── Step 6: Initialize NLP Components ──
    logger.info("\n🧠 Step 6: Initializing NLP components...")
    from src.nlp.feature_extractor import FeatureExtractor
    from src.nlp.commitment_detector import CommitmentDetector
    from src.nlp.risk_detector import RiskDetector
    from src.nlp.urgency_detector import UrgencyImportanceDetector
    from src.nlp.summarizer import EmailSummarizer

    feature_extractor = FeatureExtractor()
    commitment_detector = CommitmentDetector()
    risk_detector = RiskDetector()
    urgency_detector = UrgencyImportanceDetector()
    summarizer = EmailSummarizer()
    logger.info("NLP components initialized")

    # ── Step 7: Split Data ──
    logger.info("\n✂️ Step 7: Splitting data...")
    split = create_train_val_test_split(annotated_emails, strategy='by_sender')
    logger.info(f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}")

    # ── Step 8: Train Models ──
    logger.info("\n🏋️ Step 8: Training priority classifiers...")
    from src.ml.embeddings import EmbeddingManager
    from src.ml.trainer import ModelTrainer

    embedding_mgr = EmbeddingManager()
    trainer = ModelTrainer(db, feature_extractor, embedding_mgr)
    train_results = trainer.train_priority_model_from_emails(
        split['train'], split['val'], split['test']
    )
    logger.info(f"Best model: {train_results.get('best_model', 'unknown')}")

    # ── Step 9: Evaluate ──
    logger.info("\n📊 Step 9: Evaluating on test set...")
    eval_results = trainer.evaluate_all_models_from_emails(split['test'])
    logger.info(f"Best test F1: {eval_results.get('best_f1', 0):.4f}")

    # Cross-validation
    logger.info("Running 5-fold cross-validation...")
    cv_results = trainer.run_cross_validation(annotated_emails[:1000], model_name='logistic_regression', k=5)
    logger.info(f"CV mean F1: {cv_results.get('mean_f1', 0):.4f} ± {cv_results.get('std_f1', 0):.4f}")

    # Save results
    results_path = MODELS_DIR / "experiment_results.json"
    trainer.save_results(str(results_path))
    logger.info(f"Results saved to {results_path}")

    # ── Step 10: Compute Embeddings ──
    logger.info("\n🔢 Step 10: Computing email embeddings...")
    texts = [e.get('cleaned_body', e.get('body', ''))[:500] for e in annotated_emails]
    embeddings = embedding_mgr.encode(texts, show_progress=True)
    emb_path = MODELS_DIR / "email_embeddings.npy"
    embedding_mgr.save_embeddings(embeddings, str(emb_path))
    logger.info(f"Embeddings saved (shape: {embeddings.shape})")

    # ── Step 11: Process Intelligence ──
    logger.info("\n🔍 Step 11: Processing intelligence modules...")

    # Priority scoring (score first 500 for efficiency)
    from src.intelligence.priority_engine import PriorityEngine
    priority_engine = PriorityEngine(db)
    scored = 0
    for email in annotated_emails[:500]:
        try:
            priority_engine.score_email(email)
            scored += 1
        except Exception as e:
            logger.debug(f"Score error: {e}")
    logger.info(f"Priority scored {scored} emails")

    # Store annotations
    logger.info("Storing annotations in database...")
    ann_stored = 0
    for email in annotated_emails:
        ann = email.get('annotations')
        eid = email.get('id') or email.get('db_id') or email.get('email_id')
        if ann and eid:
            try:
                db.add_annotation(eid, ann)
                ann_stored += 1
            except Exception:
                pass
    logger.info(f"Stored {ann_stored} annotations")

    # Conversation threads
    from src.intelligence.intelligence_modules import (
        ConversationMemory, CommitmentTracker, RiskMonitor, RelationshipIntelligence, DailyBriefingGenerator
    )
    conv_memory = ConversationMemory(db, embedding_mgr)
    threads = conv_memory.reconstruct_threads(annotated_emails)
    logger.info(f"Reconstructed {len(threads)} threads")

    # Persist threads in DB
    for tid, t_emails in threads.items():
        try:
            db.upsert_thread({
                'thread_id': tid,
                'subject': t_emails[0].get('subject', '') if t_emails else '',
                'email_count': len(t_emails),
                'participants': json.dumps(list(set(e.get('sender', '') for e in t_emails if e.get('sender')))),
                'status': 'active',
            })
        except Exception:
            pass

    # Commitments
    commitment_tracker = CommitmentTracker(db, commitment_detector)
    c_count = 0
    for email in annotated_emails[:500]:
        try:
            cs = commitment_tracker.process_email_commitments(email)
            c_count += len(cs)
        except Exception as e:
            logger.debug(f"Commitment error: {e}")
    logger.info(f"Detected {c_count} commitments")

    # Risks
    risk_monitor = RiskMonitor(db, risk_detector)
    r_count = 0
    for email in annotated_emails[:500]:
        try:
            rs = risk_monitor.process_email_risks(email)
            r_count += len(rs)
        except Exception:
            pass
    logger.info(f"Detected {r_count} risks")

    # Contacts / Relationship intelligence
    rel_intel = RelationshipIntelligence(db)
    try:
        # Build contacts from emails
        for email in annotated_emails:
            sender = email.get('sender', '')
            if sender:
                try:
                    db.update_contact_stats(sender, sent=True)
                except Exception:
                    try:
                        name = email.get('sender_name', sender.split('@')[0])
                        org = sender.split('@')[-1].split('.')[0] if '@' in sender else ''
                        db.add_contact({'email_address': sender, 'name': name, 'organization': org})
                    except Exception:
                        pass
        rel_intel.build_communication_graph()
        logger.info("Communication graph built")
    except Exception as e:
        logger.warning(f"Relationship graph error: {e}")

    # Generate initial Daily Briefing
    try:
        dbg = DailyBriefingGenerator(db, priority_engine, commitment_tracker, risk_monitor)
        briefing = dbg.generate_briefing()
        logger.info(f"Generated initial daily briefing for {briefing.get('date')}")
    except Exception as e:
        logger.warning(f"Briefing generation error: {e}")

    # ── Step 12: Final Summary ──
    elapsed = (datetime.now() - start_time).total_seconds()
    final_stats = db.get_stats()

    logger.info("\n" + "=" * 60)
    logger.info("📋 TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Time elapsed: {elapsed:.1f} seconds")
    logger.info(f"Emails: {final_stats.get('email_count', 0)}")
    logger.info(f"Threads: {len(threads)}")
    logger.info(f"Commitments: {c_count}")
    logger.info(f"Risks: {r_count}")
    logger.info(f"Best model: {eval_results.get('best_model', 'unknown')}")
    logger.info(f"Best F1: {eval_results.get('best_f1', 0):.4f}")
    logger.info(f"\nRun dashboard: streamlit run dashboard/app.py")

    # Save training summary
    training_summary = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'emails_processed': final_stats.get('email_count', 0),
        'threads': len(threads),
        'commitments': c_count,
        'risks': r_count,
        'best_model': eval_results.get('best_model', 'unknown'),
        'best_f1': eval_results.get('best_f1', 0),
        'evaluation_results': eval_results,
    }
    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
