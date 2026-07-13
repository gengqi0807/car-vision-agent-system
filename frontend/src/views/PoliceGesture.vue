<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>交警手势识别</h1>
      <p>实时视频流分析，识别 8 种中国标准交警指挥手势</p>
    </header>

    <section class="two-col">
      <article class="panel">
        <div class="video-input-row">
          <input v-model="videoUrl" type="text" placeholder="输入视频流地址 (rtmp / hls / mp4)..." />
          <button type="button" class="btn-video" @click="connected = true">开启识别</button>
          <button type="button" class="btn-video secondary" @click="connected = false">停止</button>
        </div>
        <div class="video-status">
          <span v-if="!connected" class="off">未连接</span>
          <span v-else>已连接 · 正在拉流...</span>
        </div>

        <div class="image-placeholder police-frame">
          交警手势视频流
          <div class="small">人体关键点与手势标签将在这里实时叠加</div>
        </div>

        <div>
          <span class="gesture-tag">左转弯信号</span>
          <span class="gesture-confidence">置信度 92.3%</span>
        </div>
        <div class="stream-meta">帧率 25fps · 识别中</div>
      </article>

      <article class="panel">
        <h4>识别结果</h4>
        <div class="result-item">
          <span>左转弯信号</span>
          <span class="val">92.3%</span>
        </div>
        <div class="result-item inactive">
          <span>直行信号</span>
          <span class="val">45.1%</span>
        </div>
        <div class="result-item inactive">
          <span>停止信号</span>
          <span class="val">28.7%</span>
        </div>
        <div class="result-item inactive">
          <span>右转弯信号</span>
          <span class="val">12.4%</span>
        </div>
        <div class="support-label">支持 8 种标准手势</div>
        <div class="support-tags">停止 · 直行 · 左转弯 · 左待转 · 右转弯 · 变道 · 减速 · 靠边</div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import { ref } from "vue";

const videoUrl = ref("");
const connected = ref(false);
</script>

<style scoped lang="scss">
.police-frame {
  height: 220px;
  margin-top: 12px;
}

.stream-meta {
  margin-top: 6px;
  font-size: 13px;
  color: var(--muted-soft);
}
</style>
