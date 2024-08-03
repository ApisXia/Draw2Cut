<template>
    <PointCloud />
</template>

<script>
import PointCloud from './components/PointCloud.vue';

export default {
  name: 'App',
  components: {
    PointCloud
  }
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  box-sizing: border-box;
  color: #2c3e50;
  padding: 0px 60px;
  height: 100vh;
  display: flex;
}
</style>
