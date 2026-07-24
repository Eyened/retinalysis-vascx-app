import './styles.css';
import { mount } from 'svelte';
import App from './App.svelte';
import UnsupportedDevice from '$lib/components/UnsupportedDevice.svelte';
import { isUnsupportedDevice } from '$lib/utils/device';

const target = document.getElementById('app') as HTMLElement;
const app = mount(isUnsupportedDevice() ? UnsupportedDevice : App, { target });

export default app;
