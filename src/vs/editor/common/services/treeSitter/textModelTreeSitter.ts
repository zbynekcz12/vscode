/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import type * as Parser from '@vscode/tree-sitter-wasm';
import { ITreeSitterParseResult, ITextModelTreeSitter, RangeChange, TreeParseUpdateEvent, ITreeSitterImporter, ModelTreeUpdateEvent } from '../treeSitterParserService.js';
import { Disposable, DisposableStore, IDisposable } from '../../../../base/common/lifecycle.js';
import { ITextModel } from '../../model.js';
import { IModelContentChange, IModelContentChangedEvent } from '../../textModelEvents.js';
import { ITelemetryService } from '../../../../platform/telemetry/common/telemetry.js';
import { ILogService } from '../../../../platform/log/common/log.js';
import { setTimeout0 } from '../../../../base/common/platform.js';
import { Emitter, Event } from '../../../../base/common/event.js';
import { cancelOnDispose } from '../../../../base/common/cancellation.js';
import { Range } from '../../core/range.js';
import { Position } from '../../core/position.js';
import { LimitedQueue } from '../../../../base/common/async.js';
import { TextLength } from '../../core/textLength.js';
import { TreeSitterLanguages } from './treeSitterLanguages.js';
import { AppResourcePath, FileAccess } from '../../../../base/common/network.js';
import { IFileService } from '../../../../platform/files/common/files.js';

interface ChangedRange {
	newNodeId: number;
	newStartPosition: Position;
	newEndPosition: Position;
	newStartIndex: number;
	newEndIndex: number;
	oldStartIndex: number;
	oldEndIndex: number;
}

export interface TextModelTreeSitterItem {
	dispose(): void;
	textModelTreeSitter: TextModelTreeSitter;
	disposables: DisposableStore;
}

const enum TelemetryParseType {
	Full = 'fullParse',
	Incremental = 'incrementalParse'
}


export class TextModelTreeSitter extends Disposable implements ITextModelTreeSitter {
	private _onDidChangeParseResult: Emitter<ModelTreeUpdateEvent> = this._register(new Emitter<ModelTreeUpdateEvent>());
	public readonly onDidChangeParseResult: Event<ModelTreeUpdateEvent> = this._onDidChangeParseResult.event;
	private _rootTreeSitterTree: TreeSitterParseResult | undefined;

	private _query: Parser.Query | undefined;
	private _injectedTreeSitterTrees: Map<string, TreeSitterParseResult> = new Map();
	private _versionId: number = 0;

	get parseResult(): ITreeSitterParseResult | undefined { return this._rootTreeSitterTree; }

	constructor(
		readonly model: ITextModel,
		private readonly _treeSitterLanguages: TreeSitterLanguages,
		parseImmediately: boolean = true,
		@ITreeSitterImporter private readonly _treeSitterImporter: ITreeSitterImporter,
		@ILogService private readonly _logService: ILogService,
		@ITelemetryService private readonly _telemetryService: ITelemetryService,
		@IFileService private readonly _fileService: IFileService
	) {
		super();
		if (parseImmediately) {
			this._register(Event.runAndSubscribe(this.model.onDidChangeLanguage, (e => this._onDidChangeLanguage(e ? e.newLanguage : this.model.getLanguageId()))));
		} else {
			this._register(this.model.onDidChangeLanguage(e => this._onDidChangeLanguage(e ? e.newLanguage : this.model.getLanguageId())));
		}
	}

	private readonly _parseSessionDisposables = this._register(new DisposableStore());
	/**
	 * Be very careful when making changes to this method as it is easy to introduce race conditions.
	 */
	private async _onDidChangeLanguage(languageId: string) {
		this.parse(languageId);
	}

	public async parse(languageId: string = this.model.getLanguageId()): Promise<ITreeSitterParseResult | undefined> {
		this._parseSessionDisposables.clear();
		this._rootTreeSitterTree = undefined;

		const token = cancelOnDispose(this._parseSessionDisposables);
		const language = await this._treeSitterLanguages.getLanguage(languageId);
		if (!language) {
			return;
		}

		const Parser = await this._treeSitterImporter.getParserClass();
		if (token.isCancellationRequested) {
			return;
		}

		const treeSitterTree = this._parseSessionDisposables.add(new TreeSitterParseResult(new Parser(), languageId, language, this._logService, this._telemetryService));
		this._rootTreeSitterTree = treeSitterTree;
		this._parseSessionDisposables.add(treeSitterTree.onDidUpdate(e => this._handleTreeUpdate(e)));
		this._parseSessionDisposables.add(this.model.onDidChangeContent(e => this._onDidChangeContent(treeSitterTree, e)));
		this._onDidChangeContent(treeSitterTree, undefined);
		if (token.isCancellationRequested) {
			return;
		}

		return this._rootTreeSitterTree;
	}

	private _handleTreeUpdate(e: TreeParseUpdateEvent) {
		if (e.ranges && (e.versionId > this._versionId)) {
			this._versionId = e.versionId;
			// kick off check for injected languages
			this._parseInjected();

			const ranges: Record<string, RangeChange[]> = {};
			ranges[e.language] = e.ranges;
			this._onDidChangeParseResult.fire({ ranges, versionId: e.versionId });
		}
	}

	private _queries: string | undefined;
	private async _ensureInjectionQueries() {
		if (!this._queries) {
			const injectionsQueriesLocation: AppResourcePath = `vs/editor/common/languages/injections/${this.model.getLanguageId()}.scm`;
			const uri = FileAccess.asFileUri(injectionsQueriesLocation);
			if (!(await this._fileService.exists(uri))) {
				this._queries = '';
			} else {
				const query = await this._fileService.readFile(uri);
				this._queries = query.value.toString();
			}
		}
		return this._queries;
	}

	private async _getQuery() {
		if (!this._query) {
			const language = await this._treeSitterLanguages.getLanguage(this.model.getLanguageId());
			if (!language) {
				return;
			}
			const queries = await this._ensureInjectionQueries();
			if (queries === '') {
				return;
			}
			const Query = await this._treeSitterImporter.getQueryClass();
			this._query = new Query(language, queries);
		}
		return this._query;
	}

	private async _parseInjected() {
		const tree = this._rootTreeSitterTree?.tree;
		if (!tree) {
			return;
		}
		const query = await this._getQuery();
		if (!query) {
			return;
		}

		const injectionCaptures = query.captures(tree.rootNode);

		// TODO @alexr00: Use a better data structure for this
		const injections: Map<string, Parser.Range[]> = new Map();
		for (const capture of injectionCaptures) {
			const injectionLanguage = capture.setProperties ? capture.setProperties['injection.language'] : undefined;
			if (injectionLanguage) {
				const range: Parser.Range = capture.node;
				if (!injections.has(injectionLanguage)) {
					injections.set(injectionLanguage, []);
				}
				injections.get(injectionLanguage)?.push(range);
			}
		}
		for (const [languageId, ranges] of injections) {
			const language = await this._treeSitterLanguages.getLanguage(languageId);
			if (!language) {
				continue;
			}
			let treeSitterTree = this._injectedTreeSitterTrees.get(languageId);
			if (!treeSitterTree) {
				const Parser = await this._treeSitterImporter.getParserClass();
				treeSitterTree = new TreeSitterParseResult(new Parser(), languageId, language, this._logService, this._telemetryService);
				this._parseSessionDisposables.add(treeSitterTree.onDidUpdate(e => this._handleTreeUpdate(e)));
				this._injectedTreeSitterTrees.set(languageId, treeSitterTree);
			}
			treeSitterTree.ranges = ranges;
			this._onDidChangeContent(treeSitterTree, undefined);
		}
	}

	private _onDidChangeContent(treeSitterTree: TreeSitterParseResult, change: IModelContentChangedEvent | undefined) {
		return treeSitterTree.onDidChangeContent(this.model, change);
	}
}


export class TreeSitterParseResult implements IDisposable, ITreeSitterParseResult {
	private _tree: Parser.Tree | undefined;
	private _lastFullyParsed: Parser.Tree | undefined;
	private _lastFullyParsedWithEdits: Parser.Tree | undefined;
	private readonly _onDidUpdate: Emitter<TreeParseUpdateEvent> = new Emitter<TreeParseUpdateEvent>();
	public readonly onDidUpdate: Event<TreeParseUpdateEvent> = this._onDidUpdate.event;
	private _versionId: number = 0;
	private _editVersion: number = 0;
	get versionId() {
		return this._versionId;
	}
	private _isDisposed: boolean = false;
	constructor(public readonly parser: Parser.Parser,
		private readonly _language: string,
		public /** exposed for tests **/ readonly language: Parser.Language,
		private readonly _logService: ILogService,
		private readonly _telemetryService: ITelemetryService) {
		this.parser.setLanguage(language);
	}
	dispose(): void {
		this._isDisposed = true;
		this._onDidUpdate.dispose();
		this._tree?.delete();
		this._lastFullyParsed?.delete();
		this._lastFullyParsedWithEdits?.delete();
		this.parser?.delete();
	}
	get tree() { return this._lastFullyParsed; }
	get isDisposed() { return this._isDisposed; }

	private findChangedNodes(newTree: Parser.Tree, oldTree: Parser.Tree): ChangedRange[] {
		const newCursor = newTree.walk();
		const oldCursor = oldTree.walk();
		const gotoNextSibling = () => {
			const n = newCursor.gotoNextSibling();
			const o = oldCursor.gotoNextSibling();
			if (n !== o) {
				throw new Error('Trees are out of sync');
			}
			return n && o;
		};
		const gotoParent = () => {
			const n = newCursor.gotoParent();
			const o = oldCursor.gotoParent();
			if (n !== o) {
				throw new Error('Trees are out of sync');
			}
			return n && o;
		};
		const gotoNthChild = (index: number) => {
			const n = newCursor.gotoFirstChild();
			const o = oldCursor.gotoFirstChild();
			if (n !== o) {
				throw new Error('Trees are out of sync');
			}
			if (index === 0) {
				return n && o;
			}
			for (let i = 1; i <= index; i++) {
				const nn = newCursor.gotoNextSibling();
				const oo = oldCursor.gotoNextSibling();
				if (nn !== oo) {
					throw new Error('Trees are out of sync');
				}
				if (!nn || !oo) {
					return false;
				}
			}
			return n && o;
		};

		const changedRanges: ChangedRange[] = [];
		let next = true;
		const nextSiblingOrParentSibling = () => {
			do {
				if (newCursor.currentNode.nextSibling) {
					return gotoNextSibling();
				}
				if (newCursor.currentNode.parent) {
					gotoParent();
				}
			} while (newCursor.currentNode.nextSibling || newCursor.currentNode.parent);
			return false;
		};

		const getClosestPreviousNodes = (): { old: Parser.Node; new: Parser.Node } | undefined => {
			// Go up parents until the end of the parent is before the start of the current.
			const newFindPrev = newTree.walk();
			newFindPrev.resetTo(newCursor);
			const oldFindPrev = oldTree.walk();
			oldFindPrev.resetTo(oldCursor);
			const startingNode = newCursor.currentNode;
			do {
				if (newFindPrev.currentNode.previousSibling && ((newFindPrev.currentNode.endIndex - newFindPrev.currentNode.startIndex) !== 0)) {
					newFindPrev.gotoPreviousSibling();
					oldFindPrev.gotoPreviousSibling();
				} else {
					while (!newFindPrev.currentNode.previousSibling && newFindPrev.currentNode.parent) {
						newFindPrev.gotoParent();
						oldFindPrev.gotoParent();
					}
					newFindPrev.gotoPreviousSibling();
					oldFindPrev.gotoPreviousSibling();
				}
			} while ((newFindPrev.currentNode.endIndex > startingNode.startIndex)
			&& (newFindPrev.currentNode.parent || newFindPrev.currentNode.previousSibling)

				&& (newFindPrev.currentNode.id !== startingNode.id));

			if ((newFindPrev.currentNode.id !== startingNode.id) && newFindPrev.currentNode.endIndex <= startingNode.startIndex) {
				return { old: oldFindPrev.currentNode, new: newFindPrev.currentNode };
			} else {
				return undefined;
			}
		};
		do {
			if (newCursor.currentNode.hasChanges) {
				// Check if only one of the children has changes.
				// If it's only one, then we go to that child.
				// If it's more then, we need to go to each child
				// If it's none, then we've found one of our ranges
				const newChildren = newCursor.currentNode.children;
				const indexChangedChildren: number[] = [];
				const changedChildren = newChildren.filter((c, index) => {
					if (c?.hasChanges) {
						indexChangedChildren.push(index);
					}
					return c?.hasChanges;
				});
				// If we have changes and we *had* an error, the whole node should be refreshed.
				if ((changedChildren.length === 0) || oldCursor.currentNode.hasError) {
					// walk up again until we get to the first one that's named as unnamed nodes can be too granular
					while (newCursor.currentNode.parent && !newCursor.currentNode.isNamed && next) {
						next = gotoParent();
					}

					const newNode = newCursor.currentNode;
					const oldNode = oldCursor.currentNode;

					const newEndPosition = new Position(newNode.endPosition.row + 1, newNode.endPosition.column + 1);
					const oldEndIndex = oldNode.endIndex;

					// Fill holes between nodes.
					const closestPrev = getClosestPreviousNodes();
					const newStartPosition = new Position(closestPrev ? closestPrev.new.endPosition.row + 1 : newNode.startPosition.row + 1, closestPrev ? closestPrev.new.endPosition.column + 1 : newNode.startPosition.column + 1);
					const newStartIndex = closestPrev ? closestPrev.new.endIndex : newNode.startIndex;
					const oldStartIndex = closestPrev ? closestPrev.old.endIndex : oldNode.startIndex;

					changedRanges.push({ newStartPosition, newEndPosition, oldStartIndex, oldEndIndex, newNodeId: newNode.id, newStartIndex, newEndIndex: newNode.endIndex });
					next = nextSiblingOrParentSibling();
				} else if (changedChildren.length >= 1) {
					next = gotoNthChild(indexChangedChildren[0]);
				}
			} else {
				next = nextSiblingOrParentSibling();
			}
		} while (next);

		if (changedRanges.length === 0 && newTree.rootNode.hasChanges) {
			return [{ newStartPosition: new Position(newTree.rootNode.startPosition.row + 1, newTree.rootNode.startPosition.column + 1), newEndPosition: new Position(newTree.rootNode.endPosition.row + 1, newTree.rootNode.endPosition.column + 1), oldStartIndex: oldTree.rootNode.startIndex, oldEndIndex: oldTree.rootNode.endIndex, newStartIndex: newTree.rootNode.startIndex, newEndIndex: newTree.rootNode.endIndex, newNodeId: newTree.rootNode.id }];
		} else {
			return changedRanges;
		}
	}

	private calculateRangeChange(changedNodes: ChangedRange[] | undefined): RangeChange[] | undefined {
		if (!changedNodes) {
			return undefined;
		}

		// Collapse conginguous ranges
		const ranges: RangeChange[] = [];
		for (let i = 0; i < changedNodes.length; i++) {
			const node = changedNodes[i];

			// Check if contiguous with previous
			const prevNode = changedNodes[i - 1];
			if ((i > 0) && prevNode.newEndPosition.equals(node.newStartPosition)) {
				const prevRangeChange = ranges[ranges.length - 1];
				prevRangeChange.newRange = new Range(prevRangeChange.newRange.startLineNumber, prevRangeChange.newRange.startColumn, node.newEndPosition.lineNumber, node.newEndPosition.column);
				prevRangeChange.oldRangeLength = node.oldEndIndex - prevNode.oldStartIndex;
				prevRangeChange.newRangeEndOffset = node.newEndIndex;
			} else {
				ranges.push({ newRange: Range.fromPositions(node.newStartPosition, node.newEndPosition), oldRangeLength: node.oldEndIndex - node.oldStartIndex, newRangeStartOffset: node.newStartIndex, newRangeEndOffset: node.newEndIndex });
			}
		}
		return ranges;
	}

	private _onDidChangeContentQueue: LimitedQueue = new LimitedQueue();
	public onDidChangeContent(model: ITextModel, changes: IModelContentChangedEvent | undefined): void {
		const version = model.getVersionId();
		if (version === this._editVersion) {
			return;
		}

		this._applyEdits(changes?.changes ?? [], version);

		this._onDidChangeContentQueue.queue(async () => {
			if (this.isDisposed) {
				// No need to continue the queue if we are disposed
				return;
			}

			let ranges: RangeChange[] | undefined;
			if (this._lastFullyParsedWithEdits && this._lastFullyParsed) {
				ranges = this.calculateRangeChange(this.findChangedNodes(this._lastFullyParsedWithEdits, this._lastFullyParsed));
			}

			const completed = await this._parseAndUpdateTree(model, version);
			if (completed) {
				if (!ranges) {
					ranges = [{ newRange: model.getFullModelRange(), oldRangeLength: model.getValueLength(), newRangeStartOffset: 0, newRangeEndOffset: model.getValueLength() }];
				}
				this._onDidUpdate.fire({ language: this._language, ranges, versionId: version });
			}
		});
	}

	private _applyEdits(changes: IModelContentChange[], version: number) {
		for (const change of changes) {
			const originalTextLength = TextLength.ofRange(Range.lift(change.range));
			const newTextLength = TextLength.ofText(change.text);
			const summedTextLengths = change.text.length === 0 ? newTextLength : originalTextLength.add(newTextLength);
			const edit = {
				startIndex: change.rangeOffset,
				oldEndIndex: change.rangeOffset + change.rangeLength,
				newEndIndex: change.rangeOffset + change.text.length,
				startPosition: { row: change.range.startLineNumber - 1, column: change.range.startColumn - 1 },
				oldEndPosition: { row: change.range.endLineNumber - 1, column: change.range.endColumn - 1 },
				newEndPosition: { row: change.range.startLineNumber + summedTextLengths.lineCount - 1, column: summedTextLengths.lineCount ? summedTextLengths.columnCount : (change.range.endColumn + summedTextLengths.columnCount) }
			};
			this._tree?.edit(edit);
			this._lastFullyParsedWithEdits?.edit(edit);
		}
		this._editVersion = version;
	}

	private async _parseAndUpdateTree(model: ITextModel, version: number): Promise<Parser.Tree | undefined> {
		const tree = await this._parse(model);
		if (tree) {
			this._tree?.delete();
			this._tree = tree;
			this._lastFullyParsed?.delete();
			this._lastFullyParsed = tree.copy();
			this._lastFullyParsedWithEdits?.delete();
			this._lastFullyParsedWithEdits = tree.copy();
			this._versionId = version;
			return tree;
		} else if (!this._tree) {
			// No tree means this is the initial parse and there were edits
			// parse function doesn't handle this well and we can end up with an incorrect tree, so we reset
			this.parser.reset();
		}
		return undefined;
	}

	private _parse(model: ITextModel): Promise<Parser.Tree | undefined> {
		let parseType: TelemetryParseType = TelemetryParseType.Full;
		if (this.tree) {
			parseType = TelemetryParseType.Incremental;
		}
		return this._parseAndYield(model, parseType);
	}

	private async _parseAndYield(model: ITextModel, parseType: TelemetryParseType): Promise<Parser.Tree | undefined> {
		let time: number = 0;
		let passes: number = 0;
		const inProgressVersion = this._editVersion;
		let newTree: Parser.Tree | null | undefined;
		this._lastYieldTime = performance.now();

		do {
			const timer = performance.now();
			try {
				newTree = this.parser.parse((index: number, position?: Parser.Point) => this._parseCallback(model, index), this._tree, { progressCallback: this._parseProgressCallback.bind(this), includedRanges: this._ranges });
			} catch (e) {
				// parsing can fail when the timeout is reached, will resume upon next loop
			} finally {
				time += performance.now() - timer;
				passes++;
			}

			// So long as this isn't the initial parse, even if the model changes and edits are applied, the tree parsing will continue correctly after the await.
			await new Promise<void>(resolve => setTimeout0(resolve));

		} while (!model.isDisposed() && !this.isDisposed && !newTree && inProgressVersion === model.getVersionId());
		this.sendParseTimeTelemetry(parseType, time, passes);
		return (newTree && (inProgressVersion === model.getVersionId())) ? newTree : undefined;
	}

	private _lastYieldTime: number = 0;
	private _parseProgressCallback(state: Parser.ParseState) {
		const now = performance.now();
		if (now - this._lastYieldTime > 50) {
			this._lastYieldTime = now;
			return true;
		}
		return false;
	}

	private _parseCallback(textModel: ITextModel, index: number): string | undefined {
		try {
			return textModel.getTextBuffer().getNearestChunk(index);
		} catch (e) {
			this._logService.debug('Error getting chunk for tree-sitter parsing', e);
		}
		return undefined;
	}

	private _ranges: Parser.Range[] | undefined;
	set ranges(ranges: Parser.Range[]) {
		if (this._ranges && ranges.length === this._ranges.length) {
			for (let i = 0; i < ranges.length; i++) {
				if (!rangesEqual(ranges[i], this._ranges[i])) {
					this._ranges = ranges;
					return;
				}
			}
		} else {
			this._ranges = ranges;
		}
	}



	private sendParseTimeTelemetry(parseType: TelemetryParseType, time: number, passes: number): void {
		this._logService.debug(`Tree parsing (${parseType}) took ${time} ms and ${passes} passes.`);
		type ParseTimeClassification = {
			owner: 'alexr00';
			comment: 'Used to understand how long it takes to parse a tree-sitter tree';
			languageId: { classification: 'SystemMetaData'; purpose: 'FeatureInsight'; comment: 'The programming language ID.' };
			time: { classification: 'SystemMetaData'; purpose: 'FeatureInsight'; isMeasurement: true; comment: 'The ms it took to parse' };
			passes: { classification: 'SystemMetaData'; purpose: 'FeatureInsight'; isMeasurement: true; comment: 'The number of passes it took to parse' };
		};
		if (parseType === TelemetryParseType.Full) {
			this._telemetryService.publicLog2<{ languageId: string; time: number; passes: number }, ParseTimeClassification>(`treeSitter.fullParse`, { languageId: this._language, time, passes });
		} else {
			this._telemetryService.publicLog2<{ languageId: string; time: number; passes: number }, ParseTimeClassification>(`treeSitter.incrementalParse`, { languageId: this._language, time, passes });
		}
	}
}

function rangesEqual(a: Parser.Range, b: Parser.Range) {
	return (a.startPosition.row === b.startPosition.row)
		&& (a.startPosition.column === b.startPosition.column)
		&& (a.endPosition.row === b.endPosition.row)
		&& (a.endPosition.column === b.endPosition.column)
		&& (a.startIndex === b.startIndex)
		&& (a.endIndex === b.endIndex);
}
