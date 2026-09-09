import "@testing-library/jest-dom/vitest";

// Vitest's jsdom environment exposes a `window` proxy whose `localStorage` /
// `sessionStorage` getters can evaluate to `undefined` (jsdom >= 26 with
// vitest 4.x). Tests that call `localStorage.clear()` / `setItem(...)` in
// `afterEach` then throw, which also skips `vi.clearAllMocks()` and leaks mock
// state across tests in the same file.
//
// Provide an in-memory fallback only when the environment's own storage is not
// usable, so working environments keep their real per-origin storage.
function createMemoryStorage() {
  let store = new Map();
  return {
    get length() {
      return store.size;
    },
    clear() {
      store = new Map();
    },
    getItem(key) {
      return store.has(String(key)) ? store.get(String(key)) : null;
    },
    key(index) {
      return [...store.keys()][index] ?? null;
    },
    removeItem(key) {
      store.delete(String(key));
    },
    setItem(key, value) {
      store.set(String(key), String(value));
    },
  };
}

function ensureWebStorage(global) {
  for (const name of ["localStorage", "sessionStorage"]) {
    let usable = false;
    try {
      usable = typeof global[name]?.setItem === "function";
    } catch {
      usable = false;
    }
    if (usable) continue;
    try {
      Object.defineProperty(global, name, {
        value: createMemoryStorage(),
        configurable: true,
        enumerable: true,
        writable: true,
      });
    } catch {
      // Non-configurable host property; fall back to assignment when allowed.
      global[name] = createMemoryStorage();
    }
  }
}

ensureWebStorage(globalThis);