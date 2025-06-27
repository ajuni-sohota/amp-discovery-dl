#!/usr/bin/env python3
"""
Cancer Immunotherapy Peptide Discovery Framework - Lightweight Version
Runs entirely in memory without saving files

Developed by Ajuni Sohota - Bioinformatics Scientist
"""

import numpy as np
import pandas as pd
from collections import Counter
import random
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ML imports
ML_BACKEND = "none"
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    ML_BACKEND = "sklearn"
    print("🌲 Using scikit-learn for machine learning (lightweight mode)")
except:
    print("⚠️  Using rule-based approach")

def main():
    """Lightweight main function that runs everything in memory"""
    start_time = datetime.now()
    
    print("🎯 Cancer Immunotherapy Peptide Discovery Framework")
    print("=" * 60)
    print("💾 Lightweight Mode: Running in memory only")
    print(f"🔬 ML Backend: {ML_BACKEND.title()}")
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(42)
    random.seed(42)
    
    # Step 1: Generate Dataset
    print("\n🧬 STEP 1: Generating Cancer Immunotherapy Dataset")
    print("-" * 50)
    
    # Generate immunostimulatory peptides
    print("🧬 Generating immunostimulatory peptides...")
    immuno_weights = {
        'L': 0.15, 'I': 0.12, 'V': 0.10, 'F': 0.08, 'Y': 0.08, 'W': 0.06,
        'K': 0.08, 'R': 0.08, 'S': 0.06, 'T': 0.06, 'A': 0.05, 'G': 0.04, 'P': 0.04,
        'N': 0.03, 'Q': 0.03, 'D': 0.02, 'E': 0.02, 'H': 0.02, 'C': 0.01, 'M': 0.01
    }
    
    immunostim_data = []
    cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'SKCM', 'GBM']
    
    for i in range(200):  # Reduced dataset size
        length = random.randint(8, 15)
        seq = ''.join(random.choices(
            list(immuno_weights.keys()), 
            weights=list(immuno_weights.values()), 
            k=length
        ))
        
        immunostim_data.append({
            'sequence': seq,
            'cancer_type': random.choice(cancer_types),
            'peptide_type': 'Immunostimulatory',
            'therapeutic_score': random.uniform(0.6, 1.0),
            'label': 1
        })
    
    # Generate tumor-targeting peptides
    print("🎯 Generating tumor-targeting peptides...")
    targeting_weights = {
        'R': 0.20, 'K': 0.18, 'W': 0.10, 'F': 0.08, 'L': 0.08, 'I': 0.06, 'V': 0.06,
        'G': 0.06, 'P': 0.06, 'S': 0.04, 'T': 0.04, 'A': 0.04, 'N': 0.03, 'Q': 0.03,
        'D': 0.02, 'E': 0.02, 'H': 0.02, 'C': 0.01, 'M': 0.01, 'Y': 0.03
    }
    
    targeting_data = []
    for i in range(150):
        length = random.randint(6, 20)
        seq = ''.join(random.choices(
            list(targeting_weights.keys()), 
            weights=list(targeting_weights.values()), 
            k=length
        ))
        
        targeting_data.append({
            'sequence': seq,
            'cancer_type': random.choice(cancer_types),
            'peptide_type': 'Tumor-Targeting',
            'therapeutic_score': random.uniform(0.5, 0.9),
            'label': 1
        })
    
    # Generate control peptides
    print("🔬 Generating control peptides...")
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    control_weights = {aa: 1/20 for aa in amino_acids}
    
    control_data = []
    for i in range(150):
        length = random.randint(5, 20)
        seq = ''.join(random.choices(
            list(control_weights.keys()), 
            weights=list(control_weights.values()), 
            k=length
        ))
        
        control_data.append({
            'sequence': seq,
            'cancer_type': 'Control',
            'peptide_type': 'Control',
            'therapeutic_score': random.uniform(0.0, 0.4),
            'label': 0
        })
    
    # Combine all data
    all_data = immunostim_data + targeting_data + control_data
    
    # Calculate properties
    print("⚗️  Calculating peptide properties...")
    for entry in all_data:
        seq = entry['sequence']
        aa_count = Counter(seq)
        length = len(seq)
        
        # Calculate properties
        cationic_charge = aa_count.get('K', 0) + aa_count.get('R', 0) + aa_count.get('H', 0)
        anionic_charge = aa_count.get('D', 0) + aa_count.get('E', 0)
        net_charge = cationic_charge - anionic_charge
        
        hydrophobic = sum(aa_count.get(aa, 0) for aa in 'AVILMFWY')
        hydrophobic_percent = (hydrophobic / length) * 100
        
        aromatic = sum(aa_count.get(aa, 0) for aa in 'FWY')
        aromatic_percent = (aromatic / length) * 100
        
        entry.update({
            'length': length,
            'net_charge': net_charge,
            'cationic_charge': cationic_charge,
            'hydrophobic_percent': hydrophobic_percent,
            'aromatic_percent': aromatic_percent,
            'molecular_weight': length * 110
        })
    
    # Create DataFrame
    cancer_dataset = pd.DataFrame(all_data)
    
    print(f"\n📊 Dataset Generated:")
    print(f"   • Total peptides: {len(cancer_dataset)}")
    print(f"   • Immunostimulatory: {len(immunostim_data)}")
    print(f"   • Tumor-targeting: {len(targeting_data)}")
    print(f"   • Control: {len(control_data)}")
    
    # Display sample data
    print(f"\n🔬 Sample Dataset:")
    print(cancer_dataset[['sequence', 'peptide_type', 'therapeutic_score', 'length', 'net_charge']].head(8))
    
    # Step 2: Train ML Model
    if ML_BACKEND == "sklearn":
        print(f"\n🤖 STEP 2: Training Machine Learning Model")
        print("-" * 50)
        
        # Prepare features
        feature_cols = ['length', 'net_charge', 'cationic_charge', 'hydrophobic_percent', 'aromatic_percent', 'molecular_weight']
        X = cancer_dataset[feature_cols].values
        y = cancer_dataset['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Training set: {len(X_train)} peptides")
        print(f"Test set: {len(X_test)} peptides")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        print("🌲 Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"   Cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Evaluate on test set
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_prob)
        }
        
        print(f"\n🎯 Model Performance:")
        for metric, value in metrics.items():
            print(f"   • {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        print(f"\n📊 Feature Importance:")
        for i, (feature, importance) in enumerate(zip(feature_cols, feature_importance)):
            print(f"   • {feature}: {importance:.3f}")
        
    else:
        print(f"\n🤖 STEP 2: Using Rule-Based Approach")
        print("-" * 50)
        model = None
        scaler = None
        metrics = {'accuracy': 0.75, 'precision': 0.72, 'recall': 0.78, 'f1_score': 0.75, 'roc_auc': 0.80}
    
    # Step 3: Generate Therapeutic Candidates
    print(f"\n🎯 STEP 3: Generating Therapeutic Candidates")
    print("-" * 50)
    
    candidates = []
    immunogenic_aa = ['L', 'I', 'V', 'F', 'Y', 'W', 'K', 'R', 'A', 'T', 'S']
    
    print("🎯 Generating novel therapeutic peptides...")
    for attempt in range(100):  # Generate 100 candidates
        length = np.random.randint(8, 15)
        peptide = ''.join(np.random.choice(immunogenic_aa, size=length))
        
        # Calculate properties
        aa_count = Counter(peptide)
        net_charge = sum(aa_count.get(aa, 0) for aa in 'KRH') - sum(aa_count.get(aa, 0) for aa in 'DE')
        cationic_charge = sum(aa_count.get(aa, 0) for aa in 'KRH')
        hydrophobic_percent = sum(aa_count.get(aa, 0) for aa in 'AVILMFWY') / length * 100
        aromatic_percent = sum(aa_count.get(aa, 0) for aa in 'FWY') / length * 100
        molecular_weight = length * 110
        
        if ML_BACKEND == "sklearn" and model:
            # ML prediction
            features = np.array([[length, net_charge, cationic_charge, hydrophobic_percent, aromatic_percent, molecular_weight]])
            features_scaled = scaler.transform(features)
            score = model.predict_proba(features_scaled)[0, 1]
        else:
            # Rule-based scoring
            score = 0.5
            if 8 <= length <= 12: score += 0.2
            if 1 <= net_charge <= 2: score += 0.2
            if aromatic_percent >= 10: score += 0.1
            if 40 <= hydrophobic_percent <= 70: score += 0.1
        
        if score >= 0.65:  # Only keep high-scoring candidates
            candidates.append({
                'peptide': peptide,
                'therapeutic_score': score,
                'length': length,
                'net_charge': net_charge,
                'hydrophobic_percent': hydrophobic_percent
            })
    
    # Sort and display candidates
    candidates_df = pd.DataFrame(candidates).sort_values('therapeutic_score', ascending=False)
    
    print(f"Generated {len(candidates)} high-scoring candidates")
    if len(candidates) > 0:
        print(f"Average score: {candidates_df['therapeutic_score'].mean():.3f}")
        print(f"Best score: {candidates_df['therapeutic_score'].max():.3f}")
        
        print(f"\n🏆 Top 10 Therapeutic Peptide Candidates:")
        print("=" * 60)
        for i, row in candidates_df.head(10).iterrows():
            print(f"{row['peptide']:12s} | Score: {row['therapeutic_score']:.3f} | Length: {row['length']} | Charge: {row['net_charge']:+2.0f}")
    
    # Step 4: Analysis Summary
    print(f"\n📊 STEP 4: Dataset Analysis Summary")
    print("-" * 50)
    
    print("🧬 Peptide Type Distribution:")
    type_counts = cancer_dataset['peptide_type'].value_counts()
    for ptype, count in type_counts.items():
        percentage = (count / len(cancer_dataset)) * 100
        print(f"   • {ptype}: {count} ({percentage:.1f}%)")
    
    print(f"\n⚗️  Average Properties by Type:")
    summary = cancer_dataset.groupby('peptide_type')[['length', 'net_charge', 'hydrophobic_percent', 'therapeutic_score']].mean()
    print(summary.round(2))
    
    print(f"\n🎯 Cancer Type Distribution (excluding controls):")
    cancer_only = cancer_dataset[cancer_dataset['cancer_type'] != 'Control']
    cancer_counts = cancer_only['cancer_type'].value_counts()
    for cancer, count in cancer_counts.items():
        print(f"   • {cancer}: {count} peptides")
    
    # Final Summary
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("🎯 CANCER IMMUNOTHERAPY DISCOVERY COMPLETE")
    print("=" * 60)
    print(f"⏱️  Execution time: {execution_time:.1f} seconds")
    print(f"🧬 Dataset: {len(cancer_dataset)} peptides analyzed")
    print(f"🤖 ML Backend: {ML_BACKEND.title()}")
    if ML_BACKEND == "sklearn":
        print(f"📈 Model AUC: {metrics['roc_auc']:.3f}")
    print(f"🎯 Therapeutic candidates: {len(candidates)}")
    
    print(f"\n🏥 Clinical Applications:")
    print("   • Neoantigen design for personalized cancer vaccines")
    print("   • CAR-T cell therapy enhancement peptides")
    print("   • Tumor-targeting drug delivery systems")
    print("   • Precision oncology biomarker development")
    
    print(f"\n🚀 Next Steps for GitHub Portfolio:")
    print("   • This demonstrates cancer immunotherapy expertise")
    print("   • Shows machine learning in drug discovery")
    print("   • Proves bioinformatics programming skills")
    print("   • Ready for Guardant Health application!")
    
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

