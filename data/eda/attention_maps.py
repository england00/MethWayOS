import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sympy import false

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Carica il file .pt
#data = torch.load(r"C:\Users\lucai\Desktop\Magistrale\Tirocinio\AIforBioinformatics\src\surv_path\checkpoints\outputs\RNA_Seq_Cross_Attention_6a186809-3422-41d0-83d2-867145830936_SurvPath_CpG_2545321_3.pt", map_location=torch.device("cuda"))
#data = torch.load(r"C:\Users\lucai\Desktop\Magistrale\Tirocinio\AIforBioinformatics\src\surv_path\checkpoints\outputs\CpG_Sites_Cross_Attention_6a186809-3422-41d0-83d2-867145830936_SurvPath_CpG_2545321_3.pt", map_location=torch.device("cuda"))
data = torch.load(r"C:\Users\lucai\Desktop\Magistrale\Tirocinio\AIforBioinformatics\src\surv_path\checkpoints\outputs\RNA_Seq_Self_Attention_6a186809-3422-41d0-83d2-867145830936_SurvPath_CpG_2545321_1.pt", map_location=torch.device("cuda"))

# === Impostazioni iniziali ===
SAVE_PATH = "attention_map.png"
VAR_THRESHOLD = 0.00045  # {0.000045, 1.00000011, 0.00045}
RNASEQ_label_list = ['GABA_B_receptor_activation', 'EGFR_interacts_with_phospholipase_C-gamma', 'Epidermal_Growth_Factor_Receptor_(EGFR)_signaling', 'CaM_pathway', 'Alternative_complement_activation', 'NADPH_regeneration', 'Synthesis_of_lysophosphatidic_acid_from_dihydroxyacetone_phosphate', 'Activation_of_CaMK_IV', 'Alpha-oxidation_of_phytanate', 'Utilization_of_Ketone_Bodies', 'Synthesis_of_Ketone_Bodies', 'Highly_calcium_permeable_nicotinic_acetylcholine_receptors', 'Lectin_pathway_of_complement_activation', 'Toll_Like_Receptor_TLR6/TLR2_Cascade', 'TRAF6_mediated_induction_of_TAK1_complex', 'Tandem_pore_domain_halothane-inhibited_K+_channel_(THIK)', 'Transferrin_endocytosis_and_recycling', 'Passive_Transport_by_Aquaporins', 'Iron_uptake_and_transport', 'Destabilization_of_mRNA_by_KSRP', 'ALK1_signaling_events', 'Vitamin_B1_(thiamin)_metabolism', 'Vitamin_B2_(riboflavin)_metabolism', 'Disinhibition_of_SNARE_formation', 'Pyrimidine_salvage_reactions', 'role_of_ran_in_mitotic_spindle_regulation', 'Regulation_of_ornithine_decarboxylase_(ODC)', 'Methionine_salvage_pathway', 'Enzymatic_degradation_of_dopamine_by_COMT', 'Degradation_of_GABA', 'Neurotransmitter_Clearance_In_The_Synaptic_Cleft', 'Enzymatic_degradation_of_Dopamine_by_monoamine_oxidase', 'Costimulation_by_the_CD28_family', 'G-protein_activation', 'Apoptotic_cleavage_of_cell_adhesion__proteins', 'TRAF6_mediated_IRF7_activation_in_TLR7/8_or_9_signaling', 'Conversion_from_APC/C/Cdc20_to_APC/C/Cdh1_in_late_anaphase', 'Highly_calcium_permeable_postsynaptic_nicotinic_acetylcholine_receptors', 'APC/C/Cdc20_mediated_degradation_of_mitotic_proteins', 'Autodegradation_of_Cdh1_by_Cdh1/APC/C', 'ahr_signal_transduction_pathway', 'Switching_of_origins_to_a_post-replicative_state', 'Polymerase_switching', 'Unwinding_of_DNA', 'Purine_catabolism', 'Purine_salvage', 'Formation_of_ATP_by_chemiosmotic_coupling', 'Purine_ribonucleoside_monophosphate_biosynthesis', 'Sulfide_oxidation_to_sulfate', 'Degradation_of_cysteine_and_homocysteine', 'Cysteine_formation_from_homocysteine', 'Sulfur_amino_acid_metabolism', 'Glyoxylate_metabolism', 'GABA_synthesis', 'ALK2_signaling_events', 'Dual_incision_reaction_in_GG-NER', 'BH3-only_proteins_associate_with_and_inactivate_anti-apoptotic_BCL-2_members', 'Activation_and_oligomerization_of_BAK_protein', 'Nectin/Necl__trans_heterodimerization', 'Serine_biosynthesis', 'Toll_Like_Receptor_3_(TLR3)_Cascade', 'activated_TAK1_mediates_p38_MAPK_activation', 'Citric_acid_cycle_(TCA_cycle)', 'Tetrahydrobiopterin_(BH4)_synthesis__recycling__salvage_and_regulation', 'NOSIP_mediated_eNOS_trafficking', 'Interconversion_of_2-oxoglutarate_and_2-hydroxyglutarate', 'Endogenous_sterols', 'Aromatic_amines_can_be_N-hydroxylated_or_N-dealkylated_by_CYP1A2', 'CYP2E1_reactions', 'Xenobiotics', 'Electric_Transmission_Across_Gap_Junctions', 'Fatty_acids', 'Eicosanoids', 'DNA_Damage_Recognition_in_GG-NER', 'Formation_of_incision_complex_in_GG-NER', 'Processing_of_DNA_ends_prior_to_end_rejoining', 'ER_Quality_Control_Compartment_(ERQC)', 'Androgen_biosynthesis', 'Basigin_interactions', 'Transport_of_connexons_to_the_plasma_membrane', 'Histamine_receptors', 'Insulin_receptor_recycling', 'Adenosine_P1_receptors', 'Removal_of_the_Flap_Intermediate', 'Bicarbonate_transporters', 'Mitochondrial_ABC_transporters', 'neuroregulin_receptor_degredation_protein-1_controls_erbb3_receptor_recycling', 'Regulation_of_Rheb_GTPase_activity_by_AMPK', 'IRAK2_mediated_activation_of_TAK1_complex', 'IRAK1_recruits_IKK_complex', 'Cross-presentation_of_soluble_exogenous_antigens_(endosomes)', 'Toll_Like_Receptor_10_(TLR10)_Cascade', 'Formation_of_transcription-coupled_NER_(TC-NER)_repair_complex', 'p53-Independent_DNA_Damage_Response', 'Insulin_effects_increased_synthesis_of_Xylulose-5-Phosphate', 'eNOS_activation', 'Monoamines_are_oxidized_to_aldehydes_by_MAOA_and_MAOB__producing_NH3_and_H2O2', 'Alcohol_catabolism', 'FMO_oxidizes_nucleophiles', 'BMP_receptor_signaling', 'PLC-mediated_hydrolysis_of_PIP2', 'Glypican_pathway', 'Stimulation_of_the_cell_death_response_by_PAK-2p34', 'Regulation_of_PAK-2p34_activity_by_PS-GAP/RHG10', 'Destabilization_of_mRNA_by_Butyrate_Response_Factor_1_(BRF1)', 'Destabilization_of_mRNA_by_Tristetraprolin_(TTP)', 'MyD88-independent_cascade_initiated_on_plasma_membrane', 'LPS_transferred_from_LBP_carrier_to_CD14', 'IKK_complex_recruitment_mediated_by_RIP1', 'Recycling_of_eIF2/GDP', 'Eukaryotic_Translation_Elongation', 'APC/C/Cdc20_mediated_degradation_of_Securin', 'Lysine_catabolism', 'Branched-chain_amino_acid_catabolism', 'Tryptophan_catabolism', 'Urea_synthesis', 'Proline_catabolism', 'Conjugation_of_benzoate_with_glycine', 'Repair_synthesis_of_patch_~27-30_bases_long__by_DNA_polymerase', 'Formation_of_the_active_cofactor__UDP-glucuronate', 'Acetylation', 'Methylation', 'Glutathione_synthesis_and_recycling', 'Conversion_of_palmitic_acid_to_very_long_chain_fatty_acyl-CoAs', 'Fatty_acid__triacylglycerol__and_ketone_body_metabolism', 'pelp1_modulation_of_estrogen_receptor_activity', 'PLCG1_events_in_ERBB2_signaling', 'GRB2/SOS_provides_linkage_to_MAPK_signaling_for_Intergrins', 'AlphaE_beta7_integrin_cell_surface_interactions', 'Synthesis_of_dolichyl-phosphate-glucose', 'Synthesis_of_GDP-mannose', 'Synthesis_of_dolichyl-phosphate_mannose', 'Biosynthesis_of_the_N-glycan_precursor_(dolichol_lipid-linked_oligosaccharide__LLO)_and_transfer_to_a_nascent_protein', 'Synthesis_of_Dolichyl-phosphate', 'Cori_Cycle_(interconversion_of_glucose_and_lactate)', 'TRIF_mediated_TLR3_signaling', 'Toll_Like_Receptor_7/8_(TLR7/8)_Cascade', 'TRAF6_mediated_induction_of_NFkB_and_MAP_kinases_upon_TLR7/8_or_9_activation', 'MyD88_dependent_cascade_initiated_on_endosome', 'IRAK2_mediated_activation_of_TAK1_complex_upon_TLR7/8_or_9_stimulation', 'IRAK1_recruits_IKK_complex_upon_TLR7/8_or_9_stimulation', 'mRNA_Decay_by_5prime_to_3prime_Exoribonuclease', 'mRNA_Decay_by_3prime_to_5prime_Exoribonuclease', 'Nicotinate_metabolism', 'Formation_of_editosomes_by_ADAR_proteins', 'Conjugation_of_salicylate_with_glycine', 'Conjugation_of_phenylacetate_with_glutamine', 'Propionyl-CoA_catabolism', 'Beta_oxidation_of_myristoyl-CoA_to_lauroyl-CoA', 'Beta_oxidation_of_palmitoyl-CoA_to_myristoyl-CoA', 'Beta_oxidation_of_decanoyl-CoA_to_octanoyl-CoA-CoA', 'Beta_oxidation_of_lauroyl-CoA_to_decanoyl-CoA-CoA', 'Beta_oxidation_of_hexanoyl-CoA_to_butanoyl-CoA', 'Beta_oxidation_of_octanoyl-CoA_to_hexanoyl-CoA', 'mitochondrial_fatty_acid_beta-oxidation_of_unsaturated_fatty_acids', 'Beta_oxidation_of_butanoyl-CoA_to_acetyl-CoA', 'Inactivation_of_APC/C_via_direct_inhibition_of_the_APC/C_complex', 'Adrenoceptors', 'Muscarinic_acetylcholine_receptors', 'Transport_to_the_Golgi_and_subsequent_modification', 'COPII_(Coat_Protein_2)_Mediated_Vesicle_Transport', 'ATM_mediated_response_to_DNA_double-strand_break', 'Serotonin_receptors', 'Activated_AMPK_stimulates_fatty-acid_oxidation_in_muscle', 'Pyrophosphate_hydrolysis', 'Eicosanoid_ligand-binding_receptors', 'Gluconeogenesis', 'Hexose_transport', 'Glycogen_synthesis', 'Glycogen_breakdown_(glycogenolysis)', 'Cdc20/Phospho-APC/C_mediated_degradation_of_Cyclin_A', 'TWIK-releated_acid-sensitive_K+_channel_(TASK)', 'PLC-gamma1_signalling', 'Signalling_to_p38_via_RIT_and_RIN', 'APC-Cdc20_mediated_degradation_of_Nek2A', 'Na+/Cl-_dependent_neurotransmitter_transporters', 'Astrocytic_Glutamate-Glutamine_Uptake_And_Metabolism', 'Metabolism_of_seratonin', 'Recognition_and_association_of_DNA_glycosylase_with_site_containing_an_affected_pyrimidine', 'Cleavage_of_the_damaged_purine', 'Recognition_and_association_of_DNA_glycosylase_with_site_containing_an_affected_purine', 'SHC_activation', 'Insulin_receptor_mediated_signaling', 'Proton/oligonucleotide_cotransporters', 'Proton-coupled_neutral_amino_acid_transporters', 'Digestion_of_dietary_lipid', 'Synthesis_of_cytosolic_5-phospho-alpha-D-ribose_1-diphosphate_(PRPP)_from_D-ribose_5-phosphate', 'Pentose_phosphate_pathway_(hexose_monophosphate_shunt)', 'Galactose_catabolism', 'Chylomicron-mediated_lipid_transport', 'Cross-presentation_of_particulate_exogenous_antigens_(phagosomes)', 'Platelet_homeostasis', 'cGMP_effects', 'The_NLRP1_inflammasome', 'Terminal_pathway_of_complement', 'Activation_of_C3_and_C5', 'The_AIM2_inflammasome', 'Gamma-carboxylation_of_protein_precursors', 'Inactivation_of_Cdc42_and_Rac', 'L1CAM_interactions', 'C6_deamination_of_adenosine', 'Cyclin_B2_mediated_events', 'Thrombin_signalling_through_proteinase_activated_receptors_(PARs)', 'ADP_signalling_through_P2Y_purinoceptor_12', 'Microtubule-dependent_trafficking_of_connexons_from_Golgi_to_the_plasma_membrane', 'Reactions_specific_to_the_complex_N-glycan_synthesis_pathway', 'Gamma-carboxylation__transport__and_amino-terminal_cleavage_of_proteins', 'DNA-PK_pathway_in_nonhomologous_end_joining', 'Transport_of_gamma-carboxylated_protein_precursors_from_the_endoplasmic_reticulum_to_the_Golgi_apparatus', 'Vasopressin-like_receptors', 'Tachykinin_receptors_bind_tachykinins', 'Inhibition_of_the_proteolytic_activity_of_APC/C_required_for_the_onset_of_anaphase_by_mitotic_spindle_checkpoint_components', 'Regulation_of_Signaling_by_NODAL', 'Polymerase_switching_on_the_C-strand_of_the_telomere', 'Removal_of_the_Flap_Intermediate_from_the_C-strand', 'Packaging_Of_Telomere_Ends', 'PERK_regulated_gene_expression', 'Activation_of_Chaperones_by_ATF6-alpha', 'Inhibition_of_HSL', 'Adrenaline_signalling_through_Alpha-2_adrenergic_receptor', 'p130Cas_linkage_to_MAPK_signaling_for_integrins', 'Arachidonate_production_from_DAG', 'Response_to_elevated_platelet_cytosolic_Ca2+', 'Axonal_growth_stimulation', 'Ceramide_signalling', 'NGF-independant_TRKA_activation', 'TRKA_activation_by_NGF', 'yaci_and_bcma_stimulation_of_b_cell_immune_responses', 'Sterols_are_12-hydroxylated_by_CYP8B1', 'Hormone_ligand-binding_receptors', 'AMPK_inhibits_chREBP_transcriptional_activation_activity', 'CaMK_IV-mediated_phosphorylation_of_CREB', 'SCF-beta-TrCP_mediated_degradation_of_Emi1', 'PKA-mediated_phosphorylation_of_CREB', 'Cam-PDE_1_activation', 'Assembly_of_the_ORC_complex_at_the_origin_of_replication', 'Displacement_of_DNA_glycosylase_by__APE1', 'Synthesis__Secretion__and_Inactivation_of_Glucose-dependent_Insulinotropic_Polypeptide_(GIP)', 'Inhibition_of_TSC_complex_formation_by_PKB', 'PDE3B_signalling', 'IRS_activation', 'Vitamins', 'PDGF_receptor_signaling_network', 'COX_reactions', 'Repair_synthesis_for_gap-filling_by_DNA_polymerase_in_TC-NER', 'Transport_of_vitamins__nucleosides__and_related_molecules', 'Transport_of_fatty_acids', 'Transport_of_nucleotide_sugars', 'N-glycan_trimming_and_elongation_in_the_cis-Golgi', 'Progressive_trimming_of_alpha-1_2-linked_mannose_residues_from_Man9/8/7GlcNAc2_to_produce_Man5GlcNAc2', 'Estrogen_biosynthesis', 'Mineralocorticoid_biosynthesis', 'Glucocorticoid_biosynthesis', 'Pregnenolone_biosynthesis', 'Synthesis_of_bile_acids_and_bile_salts_via_27-hydroxycholesterol', 'Processive_synthesis_on_the_C-strand_of_the_telomere', 'Vitamin_D_(calciferol)_metabolism', 'Activation_of_PKB', 'Proton-coupled_monocarboxylate_transport', 'Metal_ion_SLC_transporters', 'Facilitative_Na+-independent_glucose_transporters', 'Zinc_influx_into_cells_by_the_SLC39_gene_family', 'Cytosolic_tRNA_aminoacylation', 'Mitochondrial_tRNA_aminoacylation', 'Transport_of_organic_anions', 'HCN_channels', 'Inhibition__of_voltage_gated_Ca2+_channels_via_Gbeta/gamma_subunits', 'Activation_of_G_protein_gated_Potassium_channels', 'Prostanoid_metabolism', 'Synthesis_of_bile_acids_and_bile_salts', 'Leukotriene_synthesis', 'Synthesis_of_bile_acids_and_bile_salts_via_7alpha-hydroxycholesterol', 'Synthesis_of_bile_acids_and_bile_salts_via_24-hydroxycholesterol', 'CREB_phosphorylation_through_the_activation_of_Adenylate_Cyclase', 'Other_semaphorin_interactions', 'Lysosome_Vesicle_Biogenesis', 'Gap_junction_assembly', 'Sema4D_induced_cell_migration_and_growth-cone_collapse', 'reversal_of_insulin_resistance_by_leptin', 'Heme_degradation', 'Striated_Muscle_Contraction']
METH_label_list = ['GABA_B_receptor_activation', 'EGFR_interacts_with_phospholipase_C-gamma', 'Epidermal_Growth_Factor_Receptor_(EGFR)_signaling', 'CaM_pathway', 'Alternative_complement_activation', 'NADPH_regeneration', 'Synthesis_of_lysophosphatidic_acid_from_dihydroxyacetone_phosphate', 'Activation_of_CaMK_IV', 'Alpha-oxidation_of_phytanate', 'Utilization_of_Ketone_Bodies', 'Synthesis_of_Ketone_Bodies', 'Highly_calcium_permeable_nicotinic_acetylcholine_receptors', 'Lectin_pathway_of_complement_activation', 'Toll_Like_Receptor_TLR6/TLR2_Cascade', 'TRAF6_mediated_induction_of_TAK1_complex', 'Tandem_pore_domain_halothane-inhibited_K+_channel_(THIK)', 'Transferrin_endocytosis_and_recycling', 'Passive_Transport_by_Aquaporins', 'Iron_uptake_and_transport', 'Destabilization_of_mRNA_by_KSRP', 'ALK1_signaling_events', 'Vitamin_B1_(thiamin)_metabolism', 'Vitamin_B2_(riboflavin)_metabolism', 'Disinhibition_of_SNARE_formation', 'Pyrimidine_salvage_reactions', 'role_of_ran_in_mitotic_spindle_regulation', 'Regulation_of_ornithine_decarboxylase_(ODC)', 'Methionine_salvage_pathway', 'Enzymatic_degradation_of_dopamine_by_COMT', 'Degradation_of_GABA', 'Neurotransmitter_Clearance_In_The_Synaptic_Cleft', 'Enzymatic_degradation_of_Dopamine_by_monoamine_oxidase', 'Costimulation_by_the_CD28_family', 'G-protein_activation', 'Apoptotic_cleavage_of_cell_adhesion__proteins', 'TRAF6_mediated_IRF7_activation_in_TLR7/8_or_9_signaling', 'Conversion_from_APC/C/Cdc20_to_APC/C/Cdh1_in_late_anaphase', 'Highly_calcium_permeable_postsynaptic_nicotinic_acetylcholine_receptors', 'APC/C/Cdc20_mediated_degradation_of_mitotic_proteins', 'Autodegradation_of_Cdh1_by_Cdh1/APC/C', 'ahr_signal_transduction_pathway', 'Switching_of_origins_to_a_post-replicative_state', 'Polymerase_switching', 'Unwinding_of_DNA', 'Purine_catabolism', 'Purine_salvage', 'Purine_ribonucleoside_monophosphate_biosynthesis', 'Sulfide_oxidation_to_sulfate', 'Degradation_of_cysteine_and_homocysteine', 'Cysteine_formation_from_homocysteine', 'Sulfur_amino_acid_metabolism', 'Glyoxylate_metabolism', 'GABA_synthesis', 'ALK2_signaling_events', 'Dual_incision_reaction_in_GG-NER', 'BH3-only_proteins_associate_with_and_inactivate_anti-apoptotic_BCL-2_members', 'Activation_and_oligomerization_of_BAK_protein', 'Nectin/Necl__trans_heterodimerization', 'Serine_biosynthesis', 'Toll_Like_Receptor_3_(TLR3)_Cascade', 'activated_TAK1_mediates_p38_MAPK_activation', 'Citric_acid_cycle_(TCA_cycle)', 'Tetrahydrobiopterin_(BH4)_synthesis__recycling__salvage_and_regulation', 'NOSIP_mediated_eNOS_trafficking', 'Interconversion_of_2-oxoglutarate_and_2-hydroxyglutarate', 'Endogenous_sterols', 'Aromatic_amines_can_be_N-hydroxylated_or_N-dealkylated_by_CYP1A2', 'CYP2E1_reactions', 'Xenobiotics', 'Electric_Transmission_Across_Gap_Junctions', 'Fatty_acids', 'Eicosanoids', 'DNA_Damage_Recognition_in_GG-NER', 'Formation_of_incision_complex_in_GG-NER', 'Processing_of_DNA_ends_prior_to_end_rejoining', 'ER_Quality_Control_Compartment_(ERQC)', 'Androgen_biosynthesis', 'Basigin_interactions', 'Transport_of_connexons_to_the_plasma_membrane', 'Histamine_receptors', 'Insulin_receptor_recycling', 'Adenosine_P1_receptors', 'Removal_of_the_Flap_Intermediate', 'Bicarbonate_transporters', 'Mitochondrial_ABC_transporters', 'neuroregulin_receptor_degredation_protein-1_controls_erbb3_receptor_recycling', 'Regulation_of_Rheb_GTPase_activity_by_AMPK', 'IRAK2_mediated_activation_of_TAK1_complex', 'IRAK1_recruits_IKK_complex', 'Cross-presentation_of_soluble_exogenous_antigens_(endosomes)', 'Toll_Like_Receptor_10_(TLR10)_Cascade', 'Formation_of_transcription-coupled_NER_(TC-NER)_repair_complex', 'p53-Independent_DNA_Damage_Response', 'Insulin_effects_increased_synthesis_of_Xylulose-5-Phosphate', 'eNOS_activation', 'Monoamines_are_oxidized_to_aldehydes_by_MAOA_and_MAOB__producing_NH3_and_H2O2', 'Alcohol_catabolism', 'FMO_oxidizes_nucleophiles', 'BMP_receptor_signaling', 'PLC-mediated_hydrolysis_of_PIP2', 'Stimulation_of_the_cell_death_response_by_PAK-2p34', 'Regulation_of_PAK-2p34_activity_by_PS-GAP/RHG10', 'Destabilization_of_mRNA_by_Butyrate_Response_Factor_1_(BRF1)', 'Destabilization_of_mRNA_by_Tristetraprolin_(TTP)', 'MyD88-independent_cascade_initiated_on_plasma_membrane', 'LPS_transferred_from_LBP_carrier_to_CD14', 'IKK_complex_recruitment_mediated_by_RIP1', 'Recycling_of_eIF2/GDP', 'Eukaryotic_Translation_Elongation', 'APC/C/Cdc20_mediated_degradation_of_Securin', 'Lysine_catabolism', 'Branched-chain_amino_acid_catabolism', 'Tryptophan_catabolism', 'Urea_synthesis', 'Proline_catabolism', 'Conjugation_of_benzoate_with_glycine', 'Repair_synthesis_of_patch_~27-30_bases_long__by_DNA_polymerase', 'Formation_of_the_active_cofactor__UDP-glucuronate', 'Acetylation', 'Methylation', 'Glutathione_synthesis_and_recycling', 'Conversion_of_palmitic_acid_to_very_long_chain_fatty_acyl-CoAs', 'Fatty_acid__triacylglycerol__and_ketone_body_metabolism', 'pelp1_modulation_of_estrogen_receptor_activity', 'PLCG1_events_in_ERBB2_signaling', 'GRB2/SOS_provides_linkage_to_MAPK_signaling_for_Intergrins', 'AlphaE_beta7_integrin_cell_surface_interactions', 'Synthesis_of_dolichyl-phosphate-glucose', 'Synthesis_of_GDP-mannose', 'Synthesis_of_dolichyl-phosphate_mannose', 'Biosynthesis_of_the_N-glycan_precursor_(dolichol_lipid-linked_oligosaccharide__LLO)_and_transfer_to_a_nascent_protein', 'Synthesis_of_Dolichyl-phosphate', 'Cori_Cycle_(interconversion_of_glucose_and_lactate)', 'TRIF_mediated_TLR3_signaling', 'Toll_Like_Receptor_7/8_(TLR7/8)_Cascade', 'TRAF6_mediated_induction_of_NFkB_and_MAP_kinases_upon_TLR7/8_or_9_activation', 'MyD88_dependent_cascade_initiated_on_endosome', 'IRAK2_mediated_activation_of_TAK1_complex_upon_TLR7/8_or_9_stimulation', 'IRAK1_recruits_IKK_complex_upon_TLR7/8_or_9_stimulation', 'mRNA_Decay_by_5prime_to_3prime_Exoribonuclease', 'mRNA_Decay_by_3prime_to_5prime_Exoribonuclease', 'Nicotinate_metabolism', 'Formation_of_editosomes_by_ADAR_proteins', 'Conjugation_of_salicylate_with_glycine', 'Conjugation_of_phenylacetate_with_glutamine', 'Propionyl-CoA_catabolism', 'Beta_oxidation_of_myristoyl-CoA_to_lauroyl-CoA', 'Beta_oxidation_of_palmitoyl-CoA_to_myristoyl-CoA', 'Beta_oxidation_of_decanoyl-CoA_to_octanoyl-CoA-CoA', 'Beta_oxidation_of_lauroyl-CoA_to_decanoyl-CoA-CoA', 'Beta_oxidation_of_hexanoyl-CoA_to_butanoyl-CoA', 'Beta_oxidation_of_octanoyl-CoA_to_hexanoyl-CoA', 'mitochondrial_fatty_acid_beta-oxidation_of_unsaturated_fatty_acids', 'Beta_oxidation_of_butanoyl-CoA_to_acetyl-CoA', 'Inactivation_of_APC/C_via_direct_inhibition_of_the_APC/C_complex', 'Adrenoceptors', 'Muscarinic_acetylcholine_receptors', 'Transport_to_the_Golgi_and_subsequent_modification', 'COPII_(Coat_Protein_2)_Mediated_Vesicle_Transport', 'ATM_mediated_response_to_DNA_double-strand_break', 'Serotonin_receptors', 'Activated_AMPK_stimulates_fatty-acid_oxidation_in_muscle', 'Pyrophosphate_hydrolysis', 'Eicosanoid_ligand-binding_receptors', 'Gluconeogenesis', 'Hexose_transport', 'Glycogen_synthesis', 'Glycogen_breakdown_(glycogenolysis)', 'Cdc20/Phospho-APC/C_mediated_degradation_of_Cyclin_A', 'TWIK-releated_acid-sensitive_K+_channel_(TASK)', 'PLC-gamma1_signalling', 'Signalling_to_p38_via_RIT_and_RIN', 'APC-Cdc20_mediated_degradation_of_Nek2A', 'Na+/Cl-_dependent_neurotransmitter_transporters', 'Astrocytic_Glutamate-Glutamine_Uptake_And_Metabolism', 'Metabolism_of_seratonin', 'Recognition_and_association_of_DNA_glycosylase_with_site_containing_an_affected_pyrimidine', 'Cleavage_of_the_damaged_purine', 'Recognition_and_association_of_DNA_glycosylase_with_site_containing_an_affected_purine', 'SHC_activation', 'Insulin_receptor_mediated_signaling', 'Proton/oligonucleotide_cotransporters', 'Proton-coupled_neutral_amino_acid_transporters', 'Digestion_of_dietary_lipid', 'Synthesis_of_cytosolic_5-phospho-alpha-D-ribose_1-diphosphate_(PRPP)_from_D-ribose_5-phosphate', 'Pentose_phosphate_pathway_(hexose_monophosphate_shunt)', 'Galactose_catabolism', 'Chylomicron-mediated_lipid_transport', 'Cross-presentation_of_particulate_exogenous_antigens_(phagosomes)', 'cGMP_effects', 'The_NLRP1_inflammasome', 'Terminal_pathway_of_complement', 'Activation_of_C3_and_C5', 'The_AIM2_inflammasome', 'Gamma-carboxylation_of_protein_precursors', 'Inactivation_of_Cdc42_and_Rac', 'L1CAM_interactions', 'C6_deamination_of_adenosine', 'Cyclin_B2_mediated_events', 'Thrombin_signalling_through_proteinase_activated_receptors_(PARs)', 'ADP_signalling_through_P2Y_purinoceptor_12', 'Microtubule-dependent_trafficking_of_connexons_from_Golgi_to_the_plasma_membrane', 'Reactions_specific_to_the_complex_N-glycan_synthesis_pathway', 'Gamma-carboxylation__transport__and_amino-terminal_cleavage_of_proteins', 'DNA-PK_pathway_in_nonhomologous_end_joining', 'Transport_of_gamma-carboxylated_protein_precursors_from_the_endoplasmic_reticulum_to_the_Golgi_apparatus', 'Vasopressin-like_receptors', 'Tachykinin_receptors_bind_tachykinins', 'Inhibition_of_the_proteolytic_activity_of_APC/C_required_for_the_onset_of_anaphase_by_mitotic_spindle_checkpoint_components', 'Regulation_of_Signaling_by_NODAL', 'Polymerase_switching_on_the_C-strand_of_the_telomere', 'Removal_of_the_Flap_Intermediate_from_the_C-strand', 'Packaging_Of_Telomere_Ends', 'PERK_regulated_gene_expression', 'Activation_of_Chaperones_by_ATF6-alpha', 'Inhibition_of_HSL', 'Adrenaline_signalling_through_Alpha-2_adrenergic_receptor', 'p130Cas_linkage_to_MAPK_signaling_for_integrins', 'Arachidonate_production_from_DAG', 'Response_to_elevated_platelet_cytosolic_Ca2+', 'Axonal_growth_stimulation', 'Ceramide_signalling', 'NGF-independant_TRKA_activation', 'TRKA_activation_by_NGF', 'yaci_and_bcma_stimulation_of_b_cell_immune_responses', 'Sterols_are_12-hydroxylated_by_CYP8B1', 'Hormone_ligand-binding_receptors', 'AMPK_inhibits_chREBP_transcriptional_activation_activity', 'CaMK_IV-mediated_phosphorylation_of_CREB', 'SCF-beta-TrCP_mediated_degradation_of_Emi1', 'PKA-mediated_phosphorylation_of_CREB', 'Cam-PDE_1_activation', 'Assembly_of_the_ORC_complex_at_the_origin_of_replication', 'Displacement_of_DNA_glycosylase_by__APE1', 'Synthesis__Secretion__and_Inactivation_of_Glucose-dependent_Insulinotropic_Polypeptide_(GIP)', 'Inhibition_of_TSC_complex_formation_by_PKB', 'PDE3B_signalling', 'IRS_activation', 'Vitamins', 'PDGF_receptor_signaling_network', 'COX_reactions', 'Repair_synthesis_for_gap-filling_by_DNA_polymerase_in_TC-NER', 'Transport_of_vitamins__nucleosides__and_related_molecules', 'Transport_of_fatty_acids', 'Transport_of_nucleotide_sugars', 'N-glycan_trimming_and_elongation_in_the_cis-Golgi', 'Progressive_trimming_of_alpha-1_2-linked_mannose_residues_from_Man9/8/7GlcNAc2_to_produce_Man5GlcNAc2', 'Estrogen_biosynthesis', 'Mineralocorticoid_biosynthesis', 'Glucocorticoid_biosynthesis', 'Pregnenolone_biosynthesis', 'Synthesis_of_bile_acids_and_bile_salts_via_27-hydroxycholesterol', 'Processive_synthesis_on_the_C-strand_of_the_telomere', 'Vitamin_D_(calciferol)_metabolism', 'Activation_of_PKB', 'Proton-coupled_monocarboxylate_transport', 'Metal_ion_SLC_transporters', 'Facilitative_Na+-independent_glucose_transporters', 'Zinc_influx_into_cells_by_the_SLC39_gene_family', 'Cytosolic_tRNA_aminoacylation', 'Mitochondrial_tRNA_aminoacylation', 'Transport_of_organic_anions', 'HCN_channels', 'Inhibition__of_voltage_gated_Ca2+_channels_via_Gbeta/gamma_subunits', 'Activation_of_G_protein_gated_Potassium_channels', 'Prostanoid_metabolism', 'Synthesis_of_bile_acids_and_bile_salts', 'Leukotriene_synthesis', 'Synthesis_of_bile_acids_and_bile_salts_via_7alpha-hydroxycholesterol', 'Synthesis_of_bile_acids_and_bile_salts_via_24-hydroxycholesterol', 'CREB_phosphorylation_through_the_activation_of_Adenylate_Cyclase', 'Other_semaphorin_interactions', 'Lysosome_Vesicle_Biogenesis', 'Gap_junction_assembly', 'Sema4D_induced_cell_migration_and_growth-cone_collapse', 'reversal_of_insulin_resistance_by_leptin', 'Heme_degradation', 'Striated_Muscle_Contraction']

# Controlla se è tensor o dict
if isinstance(data, torch.Tensor):
    attention_map = data
elif isinstance(data, dict):
    attention_map = data.get("attn")

if attention_map is None:
    raise ValueError("Attention map not found in the loaded data.")

print("Shape of attention map:", attention_map.shape)

# Applica softmax
attention_probs = torch.nn.functional.softmax(attention_map, dim=-1)
attn_np = attention_probs.detach().cpu().numpy()


# Funzione per etichette selettive
def selective_labels(attn, threshold):
    row_std = attn.std(axis=1)
    col_std = attn.std(axis=0)

    row_labels = [f"T{i}" if row_std[i] >= threshold else "" for i in range(attn.shape[0])]
    col_labels = [f"T{i}" if col_std[i] >= threshold else "" for i in range(attn.shape[1])]
    '''
    row_sum = attn.sum(axis=1)
    col_sum = attn.sum(axis=0)
    row_labels = [f"T{i}" if row_sum[i] >= threshold else "" for i in range(attn.shape[0])]
    col_labels = [f"T{i}" if col_sum[i] >= threshold else "" for i in range(attn.shape[1])]
    '''

    return row_labels, col_labels


# Plot della mappa completa con etichette selettive
'''
def plot_attention_with_selective_labels(attn, filename, var_threshold, title="RNA_Seq Cross-Attention with CpG sites"):
    row_labels, col_labels = selective_labels(attn, var_threshold)

    final_row_labels = [a if a == '' else b for a, b in zip(row_labels, RNASEQ_label_list)]
    final_col_labels = [a if a == '' else b for a, b in zip(col_labels, METH_label_list)]

    plt.figure(figsize=(20, 17), dpi=400)  # immagine più grande
    ax = sns.heatmap(
        attn,
        cmap="magma",
        square=True,
        cbar=True,
        linewidths=0.2,
        linecolor='gray',
        xticklabels=final_col_labels,
        yticklabels=final_row_labels
    )

    # Riduci dimensione font delle etichette
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)

    plt.title(title, fontsize=14)
    plt.xlabel("Key Tokens", fontsize=10)
    plt.ylabel("Query Tokens", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    print(f"Saved attention map as {filename}")
    
def plot_attention_top_var(attn, filename, top_k=30, title="RNA_Seq Cross-Attention with CpG sites"):

    # Calcola varianza per righe e colonne
    row_var = attn.var(axis=1)
    col_var = attn.var(axis=0)

    # Indici delle top righe e colonne per varianza (ordinati in modo decrescente)
    top_row_indices = np.argsort(row_var)[-top_k:][::-1]
    top_col_indices = np.argsort(col_var)[-top_k:][::-1]

    # Filtra matrice di attenzione
    filtered_attn = attn[np.ix_(top_row_indices, top_col_indices)]

    # Etichette filtrate
    final_row_labels = [RNASEQ_label_list[i] for i in top_row_indices]
    final_col_labels = [METH_label_list[i] for i in top_col_indices]

    # Plot
    plt.figure(figsize=(20, 17), dpi=400)
    ax = sns.heatmap(
        filtered_attn,
        cmap="magma",
        square=True,
        cbar=True,
        linewidths=0.2,
        linecolor='gray',
        xticklabels=final_col_labels,
        yticklabels=final_row_labels
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)

    plt.title(title, fontsize=14)
    plt.xlabel("Top 30 CpG sites (Key Tokens)", fontsize=10)
    plt.ylabel("Top 30 Genes (Query Tokens)", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    print(f"Saved top-var attention map as {filename}")'''


def plot_attention_top_var(attn, filename, top_k=30, title="RNA-Seq Self-Attention"):
    # Calcola somma per righe e varianza per colonne
    row_var = attn.var(axis=1)
    col_var = attn.var(axis=0)

    # Trova gli indici delle top_k righe per somma
    top_row_indices = np.where(row_var >= np.partition(row_var, -top_k)[-top_k])[0]

    # Trova gli indici delle top_k colonne per varianza
    top_col_indices = np.where(col_var >= np.partition(col_var, -top_k)[-top_k])[0]

    # Seleziona 3 righe a caso tra le righe selezionate
    random_row_indices = np.random.choice(top_row_indices, size=30, replace=False)

    # Seleziona direttamente le righe e colonne senza cambiare l'ordine originale
    filtered_attn = attn[np.ix_(random_row_indices, top_col_indices)]

    # Plot
    plt.figure(figsize=(20, 17), dpi=400)
    ax = sns.heatmap(
        filtered_attn,
        cmap="magma",
        square=True,
        cbar=True,
        linewidths=0.2,
        linecolor='gray',
        xticklabels=[],
        yticklabels=[]
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)

    plt.title(title, fontsize=40)
    plt.xlabel("Key Tokens", fontsize=30)
    plt.ylabel("Query Tokens", fontsize=30)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    print(f"Saved top-var attention map as {filename}")


# === Plot ===
if attn_np.ndim == 2:
    #plot_attention_with_selective_labels(attn_np, filename=SAVE_PATH, var_threshold=VAR_THRESHOLD)
    plot_attention_top_var(attn_np, filename=SAVE_PATH)
else:
    raise ValueError("Unexpected attention map shape.")
