<template>
  <section class="page-shell">
    <header class="page-header">
      <h1>告警监控</h1>
      <p>实时告警推送 · 历史时间线 · 统计概览</p>
    </header>

    <section class="alert-stats-grid">
      <article class="alert-stat-card">
        <div class="label">总告警</div>
        <div class="number">23</div>
      </article>
      <article class="alert-stat-card">
        <div class="label">严重</div>
        <div class="number critical">3</div>
      </article>
      <article class="alert-stat-card">
        <div class="label">警告</div>
        <div class="number warning">8</div>
      </article>
      <article class="alert-stat-card">
        <div class="label">提示</div>
        <div class="number info">12</div>
      </article>
    </section>

    <section class="two-col">
      <article class="panel">
        <h4>告警时间线</h4>
        <div v-for="item in timeline" :key="item.id" class="timeline-item">
          <span class="dot" :class="item.level"></span>
          <div class="body">
            <span class="tag" :class="item.level">{{ item.levelText }}</span>
            <div class="title">{{ item.title }}</div>
            <div class="desc">{{ item.desc }}</div>
          </div>
          <span class="time">{{ item.time }}</span>
        </div>
      </article>

      <article class="panel">
        <h4>告警级别分布</h4>
        <div class="vertical-bars">
          <div class="vbar-group">
            <div class="bar critical"></div>
            <div class="bar-value">3</div>
            <div class="bar-label">严重</div>
          </div>
          <div class="vbar-group">
            <div class="bar warning"></div>
            <div class="bar-value">8</div>
            <div class="bar-label">警告</div>
          </div>
          <div class="vbar-group">
            <div class="bar info"></div>
            <div class="bar-value">12</div>
            <div class="bar-label">提示</div>
          </div>
        </div>
        <div class="chart-footer">共 23 条告警</div>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
const timeline = [
  {
    id: 1,
    level: "critical",
    levelText: "严重",
    title: "车牌识别连续失败 · LLM API 超时",
    desc: "建议检查 LLM 服务状态及网络连接",
    time: "14:23"
  },
  {
    id: 2,
    level: "warning",
    levelText: "警告",
    title: "手势识别置信度持续偏低",
    desc: "当前置信度 0.42，低于阈值 0.60",
    time: "13:15"
  },
  {
    id: 3,
    level: "info",
    levelText: "提示",
    title: "未授权访问尝试",
    desc: "来源 IP 192.168.1.45",
    time: "12:50"
  }
];
</script>
