/**
 * Mem0 provider implementations: Platform (cloud) and OSS (self-hosted).
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type {
  Mem0Config,
  Mem0Provider,
  AddOptions,
  SearchOptions,
  ListOptions,
  MemoryItem,
  AddResult,
} from "./types.ts";

// ============================================================================
// Result Normalizers
// ============================================================================

function normalizeMemoryItem(raw: any): MemoryItem {
  return {
    id: raw.id ?? raw.memory_id ?? "",
    memory: raw.memory ?? raw.text ?? raw.content ?? "",
    // Handle both platform (user_id, created_at) and OSS (userId, createdAt) field names
    user_id: raw.user_id ?? raw.userId,
    score: raw.score,
    categories: raw.categories,
    metadata: raw.metadata,
    created_at: raw.created_at ?? raw.createdAt,
    updated_at: raw.updated_at ?? raw.updatedAt,
  };
}

function normalizeSearchResults(raw: any): MemoryItem[] {
  // Platform API returns flat array, OSS returns { results: [...] }
  if (Array.isArray(raw)) return raw.map(normalizeMemoryItem);
  if (raw?.results && Array.isArray(raw.results))
    return raw.results.map(normalizeMemoryItem);
  return [];
}

function normalizeAddResult(raw: any): AddResult {
  // Handle { results: [...] } shape (both platform and OSS)
  if (raw?.results && Array.isArray(raw.results)) {
    return {
      results: raw.results.map((r: any) => ({
        id: r.id ?? r.memory_id ?? "",
        memory: r.memory ?? r.text ?? "",
        // Platform API may return PENDING status (async processing)
        // OSS stores event in metadata.event
        event: r.event ?? r.metadata?.event ?? (r.status === "PENDING" ? "ADD" : "ADD"),
      })),
    };
  }
  // Platform API without output_format returns flat array
  if (Array.isArray(raw)) {
    return {
      results: raw.map((r: any) => ({
        id: r.id ?? r.memory_id ?? "",
        memory: r.memory ?? r.text ?? "",
        event: r.event ?? r.metadata?.event ?? (r.status === "PENDING" ? "ADD" : "ADD"),
      })),
    };
  }
  return { results: [] };
}

// ============================================================================
// Platform Provider (Mem0 Cloud)
// ============================================================================

class PlatformProvider implements Mem0Provider {
  private client: any; // MemoryClient from mem0ai
  private initPromise: Promise<void> | null = null;

  constructor(
    private readonly apiKey: string,
    private readonly orgId?: string,
    private readonly projectId?: string,
    private readonly host?: string,
  ) { }

  private async ensureClient(): Promise<void> {
    if (this.client) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this._init().catch((err) => {
      this.initPromise = null;
      throw err;
    });
    return this.initPromise;
  }

  private async _init(): Promise<void> {
    const { default: MemoryClient } = await import("mem0ai");
    const opts: { apiKey: string; org_id?: string; project_id?: string; host?: string } = { apiKey: this.apiKey };
    if (this.orgId) opts.org_id = this.orgId;
    if (this.projectId) opts.project_id = this.projectId;
    if (this.host) opts.host = this.host;
    this.client = new MemoryClient(opts);
  }

  async add(
    messages: Array<{ role: string; content: string }>,
    options: AddOptions,
  ): Promise<AddResult> {
    await this.ensureClient();
    const opts: Record<string, unknown> = { user_id: options.user_id };
    if (options.run_id) opts.run_id = options.run_id;
    if (options.custom_instructions)
      opts.custom_instructions = options.custom_instructions;
    if (options.custom_categories)
      opts.custom_categories = options.custom_categories;
    if (options.enable_graph) opts.enable_graph = options.enable_graph;
    if (options.output_format) opts.output_format = options.output_format;
    if (options.source) opts.source = options.source;

    const result = await this.client.add(messages, opts);
    return normalizeAddResult(result);
  }

  async search(query: string, options: SearchOptions): Promise<MemoryItem[]> {
    await this.ensureClient();
    const filters: Record<string, unknown> = { user_id: options.user_id };
    if (options.run_id) filters.run_id = options.run_id;

    const opts: Record<string, unknown> = {
      api_version: "v2",
      filters,
    };
    if (options.top_k != null) opts.top_k = options.top_k;
    if (options.threshold != null) opts.threshold = options.threshold;
    if (options.keyword_search != null) opts.keyword_search = options.keyword_search;
    if (options.reranking != null) opts.rerank = options.reranking;

    const results = await this.client.search(query, opts);
    return normalizeSearchResults(results);
  }

  async get(memoryId: string): Promise<MemoryItem> {
    await this.ensureClient();
    const result = await this.client.get(memoryId);
    return normalizeMemoryItem(result);
  }

  async getAll(options: ListOptions): Promise<MemoryItem[]> {
    await this.ensureClient();
    const opts: Record<string, unknown> = { user_id: options.user_id };
    if (options.run_id) opts.run_id = options.run_id;
    if (options.page_size != null) opts.page_size = options.page_size;
    if (options.source) opts.source = options.source;

    const results = await this.client.getAll(opts);
    if (Array.isArray(results)) return results.map(normalizeMemoryItem);
    // Some versions return { results: [...] }
    if (results?.results && Array.isArray(results.results))
      return results.results.map(normalizeMemoryItem);
    return [];
  }

  async delete(memoryId: string): Promise<void> {
    await this.ensureClient();
    await this.client.delete(memoryId);
  }
}

// ============================================================================
// Open-Source Provider (Self-hosted)
// ============================================================================

class OSSProvider implements Mem0Provider {
  private memory: any; // Memory from mem0ai/oss
  private initPromise: Promise<void> | null = null;

  constructor(
    private readonly ossConfig?: Mem0Config["oss"],
    private readonly customPrompt?: string,
    private readonly resolvePath?: (p: string) => string,
  ) { }

  private async ensureMemory(): Promise<void> {
    if (this.memory) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this._init().catch((err) => {
      this.initPromise = null;
      throw err;
    });
    return this.initPromise;
  }

  private async _init(): Promise<void> {
    const { Memory } = await import("mem0ai/oss");

    const config: Record<string, unknown> = { version: "v1.1" };

    if (this.ossConfig?.embedder) config.embedder = this.ossConfig.embedder;
    if (this.ossConfig?.vectorStore)
      config.vectorStore = this.ossConfig.vectorStore;
    if (this.ossConfig?.llm) config.llm = this.ossConfig.llm;

    if (this.ossConfig?.historyDbPath) {
      const dbPath = this.resolvePath
        ? this.resolvePath(this.ossConfig.historyDbPath)
        : this.ossConfig.historyDbPath;
      config.historyDbPath = dbPath;
    }

    if (this.ossConfig?.disableHistory) {
      config.disableHistory = true;
    }

    if (this.customPrompt) config.customPrompt = this.customPrompt;

    try {
      this.memory = new Memory(config);
    } catch (err) {
      // If initialization fails (e.g. native SQLite binding resolution under
      // jiti), retry with history disabled — the history DB is the most common
      // source of native-binding failures and is not required for core
      // memory operations.
      if (!config.disableHistory) {
        console.warn(
          "[mem0] Memory initialization failed, retrying with history disabled:",
          err instanceof Error ? err.message : err,
        );
        config.disableHistory = true;
        this.memory = new Memory(config);
      } else {
        throw err;
      }
    }
  }

  async add(
    messages: Array<{ role: string; content: string }>,
    options: AddOptions,
  ): Promise<AddResult> {
    await this.ensureMemory();
    // OSS SDK uses camelCase: userId/runId, not user_id/run_id
    const addOpts: Record<string, unknown> = { userId: options.user_id };
    if (options.run_id) addOpts.runId = options.run_id;
    if (options.source) addOpts.source = options.source;
    const result = await this.memory.add(messages, addOpts);
    return normalizeAddResult(result);
  }

  async search(query: string, options: SearchOptions): Promise<MemoryItem[]> {
    await this.ensureMemory();
    // OSS SDK uses camelCase: userId/runId, not user_id/run_id
    const opts: Record<string, unknown> = { userId: options.user_id };
    if (options.run_id) opts.runId = options.run_id;
    if (options.limit != null) opts.limit = options.limit;
    else if (options.top_k != null) opts.limit = options.top_k;
    if (options.keyword_search != null) opts.keyword_search = options.keyword_search;
    if (options.reranking != null) opts.reranking = options.reranking;
    if (options.source) opts.source = options.source;
    if (options.threshold != null) opts.threshold = options.threshold;
    if (options.filters) opts.filters = options.filters;

    const results = await this.memory.search(query, opts);
    const normalized = normalizeSearchResults(results);

    // Filter results by threshold if specified (client-side filtering as fallback)
    if (options.threshold != null) {
      return normalized.filter(item => (item.score ?? 0) >= options.threshold!);
    }

    return normalized;
  }

  async get(memoryId: string): Promise<MemoryItem> {
    await this.ensureMemory();
    const result = await this.memory.get(memoryId);
    return normalizeMemoryItem(result);
  }

  async getAll(options: ListOptions): Promise<MemoryItem[]> {
    await this.ensureMemory();
    // OSS SDK uses camelCase: userId/runId, not user_id/run_id
    const getAllOpts: Record<string, unknown> = { userId: options.user_id };
    if (options.run_id) getAllOpts.runId = options.run_id;
    if (options.source) getAllOpts.source = options.source;
    const results = await this.memory.getAll(getAllOpts);
    if (Array.isArray(results)) return results.map(normalizeMemoryItem);
    if (results?.results && Array.isArray(results.results))
      return results.results.map(normalizeMemoryItem);
    return [];
  }

  async delete(memoryId: string): Promise<void> {
    await this.ensureMemory();
    await this.memory.delete(memoryId);
  }
}

// ============================================================================
// Self-Hosted Provider (direct HTTP, no SDK dependency)
// ============================================================================

/**
 * Provider for self-hosted mem0 REST API deployments.
 *
 * The self-hosted API differs from the platform API:
 * - No /v1/ or /v2/ path prefix (e.g. /memories, /search)
 * - Uses X-API-Key header instead of Authorization: Token
 * - No /v1/ping/ endpoint
 * - Search endpoint is /search, not /v2/memories/search/
 */
class SelfHostedProvider implements Mem0Provider {
  private readonly host: string;
  private readonly headers: Record<string, string>;

  constructor(host: string, apiKey: string) {
    this.host = host.replace(/\/+$/, ""); // strip trailing slash
    this.headers = {
      "X-API-Key": apiKey,
      "Content-Type": "application/json",
    };
  }

  private async request(path: string, options: { method: string; body?: string }): Promise<any> {
    const url = `${this.host}${path}`;
    const resp = await fetch(url, {
      method: options.method,
      headers: this.headers,
      body: options.body,
      redirect: "follow",
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`mem0 self-hosted API error ${resp.status}: ${text}`);
    }
    const contentType = resp.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      return resp.json();
    }
    return {};
  }

  async add(
    messages: Array<{ role: string; content: string }>,
    options: AddOptions,
  ): Promise<AddResult> {
    const payload: Record<string, unknown> = {
      messages,
      user_id: options.user_id,
    };
    if (options.run_id) payload.run_id = options.run_id;
    if (options.custom_instructions) payload.custom_instructions = options.custom_instructions;

    const result = await this.request("/memories", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    return normalizeAddResult(result);
  }

  async search(query: string, options: SearchOptions): Promise<MemoryItem[]> {
    const payload: Record<string, unknown> = {
      query,
      user_id: options.user_id,
    };
    if (options.top_k != null) payload.top_k = options.top_k;
    if (options.limit != null) payload.limit = options.limit;
    if (options.run_id) payload.run_id = options.run_id;
    if (options.threshold != null) payload.threshold = options.threshold;
    if (options.keyword_search != null) payload.keyword_search = options.keyword_search;
    if (options.reranking != null) payload.reranking = options.reranking;
    if (options.filters) payload.filters = options.filters;

    const result = await this.request("/search", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    return normalizeSearchResults(result);
  }

  async get(memoryId: string): Promise<MemoryItem> {
    const result = await this.request(`/memories/${memoryId}`, { method: "GET" });
    return normalizeMemoryItem(result);
  }

  async getAll(options: ListOptions): Promise<MemoryItem[]> {
    const params = new URLSearchParams();
    params.set("user_id", options.user_id);
    if (options.run_id) params.set("run_id", options.run_id);
    if (options.page_size != null) params.set("page_size", String(options.page_size));

    const result = await this.request(`/memories?${params.toString()}`, { method: "GET" });
    if (Array.isArray(result)) return result.map(normalizeMemoryItem);
    if (result?.results && Array.isArray(result.results))
      return result.results.map(normalizeMemoryItem);
    return [];
  }

  async delete(memoryId: string): Promise<void> {
    await this.request(`/memories/${memoryId}`, { method: "DELETE" });
  }
}

// ============================================================================
// Provider Factory
// ============================================================================

export function createProvider(
  cfg: Mem0Config,
  api: OpenClawPluginApi,
): Mem0Provider {
  if (cfg.mode === "open-source") {
    return new OSSProvider(cfg.oss, cfg.customPrompt, (p) =>
      api.resolvePath(p),
    );
  }

  // Self-hosted: use direct HTTP provider (no SDK, no ping, correct auth)
  if (cfg.host) {
    api.logger.info(`openclaw-mem0: using self-hosted provider at ${cfg.host}`);
    return new SelfHostedProvider(cfg.host, cfg.apiKey!);
  }

  return new PlatformProvider(cfg.apiKey!, cfg.orgId, cfg.projectId);
}
