# Import CRS projection information from /data/s2_proj_ref.json
s2proj_binary_data = importlib_resources.files(__name__).joinpath('s2_proj_ref.json')
with s2proj_binary_data.open('rb') as fp:
    byte_content = fp.read()

# Make dictionary available to modules
s2projgrid = json.loads(byte_content)

# Import OWT means for S2 MSI from /data/means_OWT_Spyrakos_S2A_B1-7.json
means_owt = importlib_resources.files(__name__).joinpath('means_OWT_Spyrakos_S2A_B1-7.json')
with means_owt.open('rb') as fp:
    byte_content = fp.read()
owts_spy_S2_B1_7 = dict(json.loads(byte_content))

# Import OWT means for S2 MSI from /data/means_OWT_Spyrakos_S2A_B2-7.json
means_owt = importlib_resources.files(__name__).joinpath('means_OWT_Spyrakos_S2A_B2-7.json')
with means_owt.open('rb') as fp:
    byte_content = fp.read()
owts_spy_S2_B2_7 = dict(json.loads(byte_content))

# Import OWT means for S2 MSI from /data/means_OWT_Cordeiro_S2A_SPM.json
means_owt = importlib_resources.files(__name__).joinpath('Means_OWT_Cordeiro_S2A_SPM.json')
with means_owt.open('rb') as fp:
    byte_content = fp.read()
owts_spm_S2_B1_8A = dict(json.loads(byte_content))

# Import OWT means for S2 MSI from /data/means_OWT_Cordeiro_S2A_SPM_B2-8A.json
means_owt = importlib_resources.files(__name__).joinpath('Means_OWT_Cordeiro_S2A_SPM_B2-8A.json')
with means_owt.open('rb') as fp:
    byte_content = fp.read()
owts_spm_S2_B2_8A = dict(json.loads(byte_content))