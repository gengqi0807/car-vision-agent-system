<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>自定义手势管理</h1>
      <p>创建、采集、训练自定义手势，通过 SVM 模型识别新类别</p>
    </header>

    <!-- 操作栏 -->
    <section class="toolbar card">
      <button class="primary-btn" @click="showCreateDialog = true">+ 创建手势</button>
      <button
        class="primary-btn"
        :disabled="trainLoading || gestures.length < 2"
        @click="handleTrain()"
        style="margin-left: 12px; background: #e8a040;"
      >
        {{ trainLoading ? '训练中...' : '训练全部手势' }}
      </button>
      <span v-if="gestures.length < 2 && gestures.length > 0" class="hint" style="margin-left: 16px;">
        至少需要 2 个手势类别才能训练
      </span>
    </section>

    <!-- 手势列表 -->
    <section class="gesture-grid" v-if="gestures.length">
      <article
        v-for="g in gestures"
        :key="g.id"
        class="card gesture-card"
        :class="{ selected: selectedGesture?.id === g.id }"
        @click="selectGesture(g)"
      >
        <div class="gesture-card-header">
          <strong class="gesture-name">{{ g.display_name || g.name }}</strong>
          <span class="badge" v-if="g.is_trained">已训练</span>
          <span class="badge muted" v-else>未训练</span>
        </div>
        <div class="gesture-meta">
          <span>{{ g.name }}</span>
          <span>{{ g.sample_count }} 样本</span>
        </div>
        <p class="gesture-desc" v-if="g.description">{{ g.description }}</p>
        <button class="danger-btn" @click.stop="confirmDelete(g)">删除</button>
      </article>
    </section>
    <section class="card empty-state" v-else>
      <p>暂无自定义手势，点击"创建手势"开始。</p>
    </section>

    <!-- 选中手势的样本区 -->
    <section class="card sample-panel" v-if="selectedGesture">
      <h2 class="section-title">
        样本列表 — {{ selectedGesture.display_name || selectedGesture.name }}
        <span class="hint">（{{ samplesTotal }} 个样本）</span>
      </h2>

      <div class="sample-actions">
        <label class="upload-btn">
          📷 上传图片采集关键点
          <input
            type="file"
            accept="image/*"
            hidden
            @change="handleImageUpload"
          />
        </label>
        <span class="hint" style="margin-left: 12px;">上传手势图片后自动提取 21 关键点</span>
      </div>

      <!-- 图片上传结果预览 -->
      <div v-if="uploadPreview" class="upload-preview card">
        <img :src="uploadPreview" alt="preview" class="preview-img" />
        <div class="preview-meta">
          <p v-if="uploadError" class="error-msg">{{ uploadError }}</p>
          <p v-else>预览已选图片，确认后将自动提取关键点并保存为样本</p>
          <button class="primary-btn" @click="confirmUploadSample" :disabled="uploadLoading">
            {{ uploadLoading ? '提交中...' : '确认为样本' }}
          </button>
          <button class="secondary-btn" @click="cancelUploadPreview">取消</button>
        </div>
      </div>

      <!-- 样本列表 -->
      <div class="samples-grid" v-if="samples.length">
        <div
          v-for="s in samples"
          :key="s.id"
          class="sample-item"
        >
          <span class="sample-id">#{{ s.id }}</span>
          <span class="sample-source">{{ s.source_type }}</span>
          <span class="sample-time">{{ formatTime(s.created_at) }}</span>
          <button class="mini-danger-btn" @click="handleDeleteSample(s.id)">删除</button>
        </div>
      </div>
      <p class="hint" v-else>暂无样本，请上传手势图片。</p>
    </section>

    <!-- 创建手势对话框 -->
    <div class="modal-overlay" v-if="showCreateDialog" @click.self="showCreateDialog = false">
      <div class="modal card">
        <h3>创建自定义手势</h3>
        <label>手势英文标识 <span class="hint">（如 peace / ok / rock，必填）</span></label>
        <input v-model="createForm.name" placeholder="peace" class="input" />
        <label style="margin-top:12px;">手势显示名称</label>
        <input v-model="createForm.display_name" placeholder="比耶" class="input" />
        <label style="margin-top:12px;">说明（可选）</label>
        <textarea v-model="createForm.description" rows="2" class="input" placeholder="描述这个手势..."></textarea>
        <div class="modal-actions">
          <button class="primary-btn" @click="handleCreate" :disabled="createLoading">
            {{ createLoading ? '创建中...' : '创建' }}
          </button>
          <button class="secondary-btn" @click="showCreateDialog = false">取消</button>
        </div>
        <p v-if="createError" class="error-msg">{{ createError }}</p>
      </div>
    </div>

    <!-- 训练结果展示 -->
    <section class="card train-result" v-if="trainResult">
      <h2 class="section-title">训练结果</h2>
      <div class="result-grid">
        <div class="result-item" :class="trainResult.status">
          <strong>状态</strong>
          <span>{{ trainResult.status === 'success' ? '✅ 成功' : '❌ 失败' }}</span>
        </div>
        <div class="result-item">
          <strong>样本数</strong><span>{{ trainResult.n_samples }}</span>
        </div>
        <div class="result-item">
          <strong>类别数</strong><span>{{ trainResult.n_classes }}</span>
        </div>
        <div class="result-item">
          <strong>手势</strong><span>{{ trainResult.class_names.join(', ') }}</span>
        </div>
      </div>
      <p>{{ trainResult.message }}</p>
      <div v-if="trainResult.evaluation && Object.keys(trainResult.evaluation).length">
        <p class="hint">交叉验证准确率: {{ trainResult.evaluation.cv_accuracy || '-' }}</p>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  fetchCustomGesturesApi,
  createCustomGestureApi,
  deleteCustomGestureApi,
  fetchCustomGestureSamplesApi,
  addCustomGestureSampleApi,
  deleteCustomGestureSampleApi,
  triggerCustomGestureTrainApi,
  type CustomGestureItem,
  type CustomGestureSampleItem,
  type CustomGestureTrainOut,
} from "@/api/owner_gesture";

// ── 手势列表 ──────────────────────────────────────────────────────
const gestures = ref<CustomGestureItem[]>([]);
const selectedGesture = ref<CustomGestureItem | null>(null);

async function loadGestures() {
  try {
    const res = await fetchCustomGesturesApi();
    gestures.value = res.data.gestures;
  } catch {
    // ignore
  }
}

function selectGesture(g: CustomGestureItem) {
  selectedGesture.value = g;
  loadSamples();
}

// ── 创建手势 ──────────────────────────────────────────────────────
const showCreateDialog = ref(false);
const createLoading = ref(false);
const createError = ref("");
const createForm = ref({ name: "", display_name: "", description: "" });

async function handleCreate() {
  createError.value = "";
  if (!createForm.value.name.trim()) {
    createError.value = "手势英文标识不能为空";
    return;
  }
  createLoading.value = true;
  try {
    await createCustomGestureApi({
      name: createForm.value.name.trim(),
      display_name: createForm.value.display_name.trim(),
      description: createForm.value.description.trim(),
    });
    createForm.value = { name: "", display_name: "", description: "" };
    showCreateDialog.value = false;
    await loadGestures();
  } catch (err: any) {
    createError.value = err?.response?.data?.detail || err.message || "创建失败";
  } finally {
    createLoading.value = false;
  }
}

// ── 删除手势 ──────────────────────────────────────────────────────
async function confirmDelete(g: CustomGestureItem) {
  if (!confirm(`确定删除手势 "${g.display_name || g.name}" 及其所有样本？`)) return;
  try {
    await deleteCustomGestureApi(g.name);
    if (selectedGesture.value?.id === g.id) {
      selectedGesture.value = null;
      samples.value = [];
    }
    await loadGestures();
  } catch {
    // ignore
  }
}

// ── 样本管理 ──────────────────────────────────────────────────────
const samples = ref<CustomGestureSampleItem[]>([]);
const samplesTotal = ref(0);

async function loadSamples() {
  if (!selectedGesture.value) return;
  try {
    const res = await fetchCustomGestureSamplesApi(selectedGesture.value.name);
    samples.value = res.data.samples;
    samplesTotal.value = res.data.total;
  } catch {
    // ignore
  }
}

async function handleDeleteSample(sampleId: number) {
  if (!confirm("确定删除此样本？")) return;
  try {
    await deleteCustomGestureSampleApi(sampleId);
    await loadSamples();
    await loadGestures();
  } catch {
    // ignore
  }
}

// ── 图片上传 + 关键点提取并入库 ──────────────────────────────────
const uploadPreview = ref<string | null>(null);
const uploadFile = ref<File | null>(null);
const uploadLoading = ref(false);
const uploadError = ref("");

function handleImageUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  uploadFile.value = file;
  uploadError.value = "";

  const reader = new FileReader();
  reader.onload = (e) => {
    uploadPreview.value = e.target?.result as string;
  };
  reader.readAsDataURL(file);

  input.value = "";
}

function cancelUploadPreview() {
  uploadPreview.value = null;
  uploadFile.value = null;
  uploadError.value = "";
}

async function confirmUploadSample() {
  if (!selectedGesture.value || !uploadFile.value) return;

  uploadLoading.value = true;
  uploadError.value = "";

  const formData = new FormData();
  formData.append("file", uploadFile.value);

  try {
    await addCustomGestureSampleApi(selectedGesture.value.name, formData);
    uploadPreview.value = null;
    uploadFile.value = null;
    await loadSamples();
    await loadGestures();
  } catch (err: any) {
    uploadError.value =
      err?.response?.data?.detail || err.message || "上传失败，请重试";
  } finally {
    uploadLoading.value = false;
  }
}

// ── 训练 ──────────────────────────────────────────────────────────
const trainLoading = ref(false);
const trainResult = ref<CustomGestureTrainOut | null>(null);

async function handleTrain() {
  trainLoading.value = true;
  trainResult.value = null;
  try {
    const res = await triggerCustomGestureTrainApi();
    trainResult.value = res.data;
    await loadGestures();
  } catch {
    trainResult.value = {
      status: "error",
      message: "训练请求失败，请检查后端日志。",
      n_samples: 0,
      n_classes: 0,
      class_names: [],
      model_path: "",
      evaluation: {},
    };
  } finally {
    trainLoading.value = false;
  }
}

// ── 工具 ──────────────────────────────────────────────────────────
function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString("zh-CN", { hour12: false });
}

onMounted(() => {
  loadGestures();
});
</script>

<style scoped lang="scss">
.toolbar {
  display: flex;
  align-items: center;
  padding: 16px 20px;
  margin-bottom: 20px;
}

.gesture-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.gesture-card {
  cursor: pointer;
  transition: border-color 0.2s;
  position: relative;

  &.selected {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent);
  }

  &:hover {
    border-color: var(--line-strong);
  }
}

.gesture-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.gesture-name {
  font-size: 16px;
}

.badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  background: #4caf50;
  color: #fff;

  &.muted {
    background: #ccc;
    color: #666;
  }
}

.gesture-meta {
  display: flex;
  gap: 12px;
  font-size: 13px;
  color: var(--muted);
  margin-bottom: 4px;
}

.gesture-desc {
  font-size: 13px;
  color: var(--text-soft);
  margin: 8px 0;
}

.danger-btn {
  font-size: 12px;
  padding: 4px 10px;
  background: #e57373;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  margin-top: 8px;

  &:hover {
    background: #d32f2f;
  }
}

.empty-state {
  text-align: center;
  padding: 40px;
  color: var(--muted);
}

.sample-panel {
  margin-top: 24px;
}

.section-title {
  font-size: 17px;
  margin-bottom: 16px;
}

.sample-actions {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
}

.upload-btn {
  display: inline-block;
  padding: 8px 16px;
  background: var(--accent);
  color: #fff;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;

  &:hover {
    opacity: 0.85;
  }
}

.upload-preview {
  display: flex;
  gap: 16px;
  align-items: center;
  margin-bottom: 16px;

  .preview-img {
    width: 160px;
    height: auto;
    border-radius: 8px;
    object-fit: contain;
    background: #f0f0f0;
  }

  .preview-meta {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
}

.samples-grid {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sample-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 12px;
  background: var(--surface);
  border-radius: 8px;
  font-size: 13px;

  .sample-id { font-weight: 600; min-width: 48px; }
  .sample-source { color: var(--muted); min-width: 80px; }
  .sample-time { flex: 1; color: var(--muted); }
}

.mini-danger-btn {
  font-size: 11px;
  padding: 2px 8px;
  background: #e57373;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;

  &:hover {
    background: #d32f2f;
  }
}

.hint {
  font-size: 13px;
  color: var(--muted);
}

/* 模态对话框 */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal {
  width: 420px;
  max-width: 92vw;
  padding: 24px;
}

.modal h3 {
  margin-bottom: 16px;
}

.modal label {
  display: block;
  font-size: 13px;
  margin-bottom: 4px;
  color: var(--text-soft);
}

.input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--line-strong);
  border-radius: 6px;
  font-size: 14px;
  background: var(--bg);
  color: var(--text);
  box-sizing: border-box;
}

.modal-actions {
  display: flex;
  gap: 10px;
  margin-top: 16px;
}

.error-msg {
  color: #e57373;
  font-size: 13px;
  margin-top: 8px;
}

/* 训练结果 */
.train-result {
  margin-top: 24px;
}

.result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}

.result-item {
  padding: 10px;
  background: var(--bg);
  border-radius: 8px;

  strong {
    display: block;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 4px;
  }

  span {
    font-size: 15px;
    font-weight: 600;
  }

  &.success span { color: #4caf50; }
  &.error span { color: #e57373; }
}

/* 复用全局的 primary-btn / secondary-btn */
.primary-btn {
  padding: 8px 18px;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;

  &:hover { opacity: 0.85; }
  &:disabled { opacity: 0.5; cursor: not-allowed; }
}

.secondary-btn {
  padding: 8px 18px;
  background: var(--border);
  color: var(--text);
  border: 1px solid var(--line-strong);
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;

  &:hover { background: var(--muted-hover); }
}
</style>
