<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>仪表盘</h1>
      <p>系统运行概览与关键指标</p>
    </header>

    <section class="stats-grid">
      <article class="stat-card card">
        <div class="label">车牌识别</div>
        <div class="number">1,284</div>
      </article>
      <article class="stat-card card">
        <div class="label">交警手势</div>
        <div class="number">856</div>
      </article>
      <article class="stat-card card">
        <div class="label">手势控车</div>
        <div class="number">2,341</div>
      </article>
      <article class="stat-card card">
        <div class="label">告警</div>
        <div class="number critical">23</div>
      </article>
    </section>

    <section class="charts-grid">
      <article class="panel">
        <h4>各模块识别趋势</h4>
        <div class="bar-chart">
          <div v-for="item in trendData" :key="item.label" class="bar-item">
            <div class="bar" :style="{ height: `${item.height}px` }"></div>
            <span class="bar-label">{{ item.label }}</span>
          </div>
        </div>
      </article>

      <article class="panel">
        <h4>最新告警</h4>
        <div v-for="alert in alerts" :key="alert.id" class="alert-item" :class="alert.level">
          <span class="dot" :class="alert.level"></span>
          <div class="content">
            <span class="tag" :class="alert.level">{{ alert.levelText }}</span>
            <div class="msg">{{ alert.message }}</div>
            <div class="desc">{{ alert.desc }}</div>
          </div>
          <span class="time">{{ alert.time }}</span>
        </div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
const trendData = [
  { label: "周一", height: 80 },
  { label: "周二", height: 110 },
  { label: "周三", height: 95 },
  { label: "周四", height: 60 },
  { label: "周五", height: 90 },
  { label: "周六", height: 125 },
  { label: "周日", height: 70 }
];

const alerts = [
  {
    id: 1,
    level: "critical",
    levelText: "严重",
    message: "车牌识别连续失败 · LLM API 超时",
    desc: "建议检查 LLM 服务状态及网络连接",
    time: "14:23"
  },
  {
    id: 2,
    level: "warning",
    levelText: "警告",
    message: "手势识别置信度持续偏低",
    desc: "当前置信度 0.42，低于阈值 0.60",
    time: "13:15"
  },
  {
    id: 3,
    level: "info",
    levelText: "提示",
    message: "未授权访问尝试",
    desc: "来源 IP 192.168.1.45",
    time: "12:50"
  }
];
</script>
