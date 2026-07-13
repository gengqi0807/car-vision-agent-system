<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>手势控车</h1>
      <p>通过手势对车辆功能进行非接触式控制</p>
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

        <div class="image-placeholder gesture-frame">
          手势识别视频流
          <div class="small">手部骨架与手势标签将在这里实时叠加</div>
        </div>

        <div>
          <span class="gesture-tag">手掌张开</span>
          <span class="gesture-confidence">置信度 95.0%</span>
        </div>
        <div class="stream-meta">帧率 30fps · 跟踪中</div>
      </article>

      <article class="panel">
        <h4>模拟控制面板</h4>
        <div class="slider-group">
          <label>音量</label>
          <div class="track"><div class="fill" style="width: 65%"></div></div>
        </div>
        <div class="slider-group">
          <label>空调温度</label>
          <div class="track"><div class="fill" style="width: 35%"></div></div>
        </div>
        <div class="btn-group">
          <button class="primary" type="button">接听电话</button>
          <button type="button">挂断电话</button>
          <button type="button">返回主页</button>
        </div>
        <div class="control-status">
          当前手势：
          <span class="highlight">手掌张开 -> 启动 / 唤醒</span>
        </div>
        <div class="support-info">
          支持 6 种手势：手掌张开 · 握拳 · 单指画圈 · 左右滑动 · 拇指向上 · 拇指向下
        </div>
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
.gesture-frame {
  height: 220px;
  margin-top: 12px;
}

.stream-meta {
  margin-top: 6px;
  font-size: 13px;
  color: var(--muted-soft);
}
</style>
