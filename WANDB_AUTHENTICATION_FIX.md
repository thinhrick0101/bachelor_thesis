# 🔑 WandB Authentication Fix Guide

## 🚨 **Problem Identified**

Your SLURM job failed with this WandB authentication error:
```
wandb: ERROR Error uploading run: returned error 401: 
{"errors":[{"message":"user is not logged in","path":["upsertBucket"],"extensions":{"code":"PERMISSION_ERROR"}}]}
```

This happens when:
- WandB API key is configured but session expired
- SLURM cluster can't access your local WandB credentials
- Network connectivity issues with WandB servers

---

## 🛠️ **Solution Steps**

### **Step 1: Fix WandB Authentication on Cluster**

```bash
# SSH to your cluster and run:
bash bachelor_thesis/fix_wandb_auth.sh
```

This will:
- ✅ Force WandB re-login with fresh credentials
- ✅ Save API key for SLURM jobs automatically
- ✅ Test the connection to ensure it works
- ✅ Set up environment variables properly

### **Step 2: Use Improved SLURM Scripts**

```bash
# Use the new SLURM script with better error handling:
bash bachelor_thesis/launch_experiments_parallel_fixed.sh
```

**Key Improvements:**
- 🔄 **Multiple Fallback Methods**: Tries saved API key, environment variable, then offline mode
- 🧪 **Connection Testing**: Tests WandB before training starts
- ⚠️ **Graceful Degradation**: Switches to offline mode if authentication fails
- 🎯 **Better Error Messages**: Clear feedback on what's happening

### **Step 3: (Optional) Add Fallback to Training Scripts**

```bash
# Add WandB error handling directly to your training scripts:
python bachelor_thesis/add_wandb_fallback.py
```

This modifies your training scripts to handle WandB errors gracefully without crashing.

---

## 🔍 **Verification**

After running the fix, verify it worked:

```bash
# Check WandB status
wandb status

# Test connection
python -c "import wandb; wandb.login(relogin=True); print('✅ WandB working!')"

# Check if API key is saved
ls -la ~/.wandb_api_key
```

---

## 🚀 **Quick Recovery**

If you need to get your experiments running **immediately**:

### **Option A: Use Offline Mode (Fastest)**
```bash
# Edit your current SLURM jobs to use offline mode
export WANDB_MODE=offline

# Re-submit jobs - they'll run without uploading to WandB
```

### **Option B: Run Without WandB**
```bash
# Temporarily disable WandB in training scripts
# Add this flag to your training commands:
--use_wandb=false
```

---

## 🎯 **Root Cause Analysis**

Your error occurred because:

1. **Session Expiry**: WandB authentication sessions expire periodically
2. **SLURM Environment**: Cluster jobs don't inherit your local WandB credentials
3. **API Key Issues**: The placeholder `"YOUR_API_KEY"` in the original script wasn't replaced

The fixed version handles all these scenarios automatically.

---

## 📊 **Impact on Your Experiments**

**✅ Good News:**
- Your dense model training likely completed successfully 
- Only the sparse model with seed 333 failed due to WandB
- You can re-run just the failed job after fixing authentication

**🔄 Recovery Steps:**
```bash
# After fixing WandB, re-run just the failed job:
sbatch sparse_seed_333_job.sh

# Or re-run all sparse models if needed:
for seed in 111 222 333; do
    sbatch sparse_seed_${seed}_job.sh
done
```

---

## 🛡️ **Prevention**

To avoid this in the future:

1. **Always use** `launch_experiments_parallel_fixed.sh` (has built-in fallbacks)
2. **Set up API key properly** using `fix_wandb_auth.sh`
3. **Test WandB connection** before submitting large experiment batches
4. **Monitor job logs** for early error detection

---

## 🆘 **If Problems Persist**

If WandB authentication continues to fail:

1. **Use offline mode** for immediate results
2. **Contact your cluster admin** about WandB access
3. **Run locally** for smaller experiments
4. **Use alternative logging** (TensorBoard, CSV files)

Your thesis experiments can continue successfully with any of these approaches! 